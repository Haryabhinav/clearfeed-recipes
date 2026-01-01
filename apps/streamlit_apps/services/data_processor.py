import re
from bs4 import BeautifulSoup

def clean_html(html_text):
    if not html_text: return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()

def get_nested_value(data, path, default=None):
    """Safely traverses nested dictionaries."""
    keys = path.split('.')
    val = data
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
    return val if val is not None else default

def construct_transcript(request):
    """
    Builds a transcript from the 'messages' array.
    """
    messages = request.get("messages", [])
    if not messages:
        desc = clean_html(request.get("description", ""))
        title = request.get("title", "")
        return f"Title: {title}\nDescription: {desc}".strip()

    # Sort by timestamp
    messages.sort(key=lambda x: x.get('ts', '0'))
    transcript = []
    
    for msg in messages[:15]: # Expanded context to 15 messages
        text = clean_html(msg.get("text", ""))
        if not text or len(text) < 2: continue
        
        # 'is_responder' is the reliable flag for Agent vs User
        role = "Agent" if msg.get("is_responder") else "User"
        transcript.append(f"{role}: {text}")
    
    full_text = "\n".join(transcript)
    if len(full_text) < 5:
        return clean_html(request.get("title", "No content provided"))
    return full_text

def format_cf_id(tickets, internal_id):
    """
    Extracts the friendly 'CF-123' key from the tickets array.
    Falls back to 'CF-{internal_id}' if no external key exists.
    """
    # 1. Try to find the 'key' in the tickets array (Official Doc location)
    if tickets and isinstance(tickets, list) and len(tickets) > 0:
        key = tickets[0].get("key")
        if key: return key
        
        # If key is missing, check 'id' in ticket object
        t_id = tickets[0].get("id")
        if t_id: return f"CF-{t_id}"

    # 2. Fallback to System ID
    if internal_id:
        return f"CF-{internal_id}"
        
    return "N/A"

def determine_source(req):
    """
    Infers source since 'source' isn't a top-level field in Request object.
    """
    # 1. Check Channel Integration Type
    c_type = get_nested_value(req, "channel.integration_type")
    if c_type: return c_type.title()

    # 2. Infer from Channel ID pattern
    c_id = get_nested_value(req, "channel.id", "")
    if c_id.startswith("C") or c_id.startswith("G"):
        return "Slack"
    if c_id.startswith("D"):
        return "Slack DM"
    
    # 3. Check for Email fields
    if req.get("author_email") and not c_id:
        return "Email"
        
    return "ClearFeed"

def process_raw_data(raw_data):
    cleaned = []
    
    for req in raw_data:
        text = construct_transcript(req)
        internal_id = str(req.get("id"))
        
        # --- 1. CF-ID (Strict Schema Match) ---
        final_cf_id = format_cf_id(req.get("tickets", []), internal_id)

        # --- 2. Channel ---
        # Docs: request['channel']['name']
        channel_name = get_nested_value(req, "channel.name")
        if not channel_name:
            channel_name = "N/A"

        # --- 3. Source ---
        source_display = determine_source(req)

        # --- 4. Author (Strict Schema Match) ---
        # Docs: 'author' is an Object containing details. 'author_email' is a String.
        author_email = req.get("author_email")
        author_obj = req.get("author", {}) # Safely handle if None
        
        # If author is just an ID string (rare but possible in old API versions)
        if isinstance(author_obj, str):
            author_id = author_obj
            author_name = None
        else:
            author_id = author_obj.get("id", "Unknown")
            author_name = author_obj.get("name") # Explicit 'name' field in object
        
        # Name Fallback Strategy
        if not author_name:
            if author_email:
                author_name = author_email.split('@')[0]
            else:
                # Deep Search in Messages (Last Resort)
                messages = req.get("messages", [])
                for msg in messages:
                    # Check if message author matches ticket author
                    m_auth = msg.get("author")
                    # msg['author'] is usually a string ID
                    if m_auth == author_id:
                        found = msg.get("user_name") or get_nested_value(msg, "user_profile.real_name")
                        if found:
                            author_name = found
                            break
        
        # Final cleanup if still unknown
        if not author_name:
            if str(author_id).startswith("U"):
                 author_name = f"Slack User {author_id}"
            else:
                 author_name = f"User {author_id}"

        if not author_email:
            author_email = "No Email"

        # --- 5. URL ---
        # Docs: request['request_thread']['url']
        url = get_nested_value(req, "request_thread.url")
        # Fallback: request['tickets'][0]['url']
        if not url:
            tickets = req.get("tickets", [])
            if tickets and len(tickets) > 0:
                url = tickets[0].get("url")
        # Fallback: Constructed URL
        if not url:
            url = f"https://app.clearfeed.ai/requests/{internal_id}"

        cleaned.append({
            "cf_id": final_cf_id,
            "request_id": internal_id,
            "channel": channel_name,
            "source": source_display,
            "author_name": author_name,
            "author_email": author_email,
            "text": text,
            "state": req.get("state"),
            "url": url, 
            "created_at": req.get("created_at")
        })
        
    return cleaned