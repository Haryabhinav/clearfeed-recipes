import re
from bs4 import BeautifulSoup

def clean_html(html_text):
    if not html_text: return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()

def get_nested_value(data, path, default=None):
    keys = path.split('.')
    val = data
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
    return val if val is not None else default

def construct_transcript(request):
    messages = request.get("messages", [])
    if not messages:
        desc = clean_html(request.get("description", ""))
        title = request.get("title", "")
        return f"Title: {title}\nDescription: {desc}".strip()
    messages.sort(key=lambda x: x.get('ts', '0'))
    transcript = []
    for msg in messages[:15]: # Expanded context to 15 messages
        text = clean_html(msg.get("text", ""))
        if not text or len(text) < 10: continue
        role = "Responder" if msg.get("is_responder") else "User"
        transcript.append(f"{role}: {text}")
    full_text = "\n".join(transcript)
    if len(full_text) >= 10:
        return full_text
    description_fallback = clean_html(request.get("description", ""))
    if len(description_fallback) >= 10:
        return description_fallback
    return clean_html(request.get("title", "No content provided"))

def determine_source(req):
    c_type = get_nested_value(req, "channel.integration_type")
    if c_type: return c_type.title()
    c_id = get_nested_value(req, "channel.id", "")
    if c_id.startswith("C") or c_id.startswith("G"):
        return "Slack"
    if c_id.startswith("D"):
        return "Slack DM"
    if req.get("author_email") and not c_id:
        return "Email"
    return "ClearFeed"

def process_raw_data(raw_data, collection_name):
    cleaned = []
    for req in raw_data:
        text = construct_transcript(req)
        internal_id = str(req.get("id"))
        channel_name = get_nested_value(req, "channel.name")
        if not channel_name:
            channel_name = "N/A (Direct/Email)"
        source_display = collection_name
        integration_type = determine_source(req)
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
        slack_url = get_nested_value(req, "request_thread.url")
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