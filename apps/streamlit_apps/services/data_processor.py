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
        role = "Responder" if msg.get("is_responder") else "User"
        transcript.append(f"{role}: {text}")
    
    full_text = "\n".join(transcript)
    if len(full_text) >= 10:
        return full_text

    # If transcript failed, try the Description
    description_fallback = clean_html(request.get("description", ""))
    if len(description_fallback) >= 10:
        return description_fallback
        
    # If both failed, return the Title (or the ultimate "No content" message)
    return clean_html(request.get("title", "No content provided"))


def determine_source(req):
    """
    Infers source since 'source' isn't a top-level field in Request object.
    Used for logic checks (like URL generation).
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

def process_raw_data(raw_data, collection_name):
    cleaned = []
    
    for req in raw_data:
        text = construct_transcript(req)
        
        # --- 1. Identifier (Internal System ID Only) ---
        internal_id = str(req.get("id"))
        
        # --- 2. Channel ---
        channel_name = get_nested_value(req, "channel.name")
        if not channel_name:
            channel_name = "N/A (Direct/Email)"

        # --- 3. Source ---
        # UPDATED: Use the passed Collection Name for display
        source_display = collection_name
        
        # Calculate actual integration type for logic (e.g. generating Slack URLs)
        integration_type = determine_source(req)

        # --- 4. Author Extraction ---
        author_email = req.get("author_email")
        author_obj = req.get("author", {}) 
        
        if isinstance(author_obj, str):
            author_id = author_obj
            author_name = None
        else:
            author_id = author_obj.get("id", "Unknown")
            author_name = author_obj.get("name")
        
        if not author_name:
            if author_email:
                author_name = author_email.split('@')[0]
            else:
                messages = req.get("messages", [])
                for msg in messages:
                    if msg.get("author") == author_id:
                        found = msg.get("user_name") or get_nested_value(msg, "user_profile.real_name")
                        if found:
                            author_name = found
                            break
        
        if not author_name:
            if str(author_id).startswith("U"):
                author_name = f"Slack User {author_id}"
            else:
                author_name = f"User {author_id}"

        if not author_email:
            author_email = "No Email"

        # --- 5. Smart URL Logic ---
        # If it's Slack and has a thread link, use it. Otherwise, use the ClearFeed Web App.
        slack_url = get_nested_value(req, "request_thread.url")
        
        # UPDATED: Check integration_type (e.g. "Slack") instead of source_display (which is now Collection Name)
        if integration_type == "Slack" and slack_url:
            url = slack_url
        else:
            url = f"https://app.clearfeed.ai/requests/{internal_id}"

        cleaned.append({
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