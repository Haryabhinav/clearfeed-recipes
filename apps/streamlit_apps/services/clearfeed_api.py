import requests
import streamlit as st
from datetime import datetime

# We use st.cache_data to prevent re-fetching if args haven't changed
@st.cache_data(ttl=600, show_spinner=False)
def get_collections(api_token):
    """Fetch all available collection IDs."""
    url = "https://api.clearfeed.app/v1/rest/collections"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("collections", [])
    except Exception as e:
        st.error(f"Error fetching collections: {e}")
        return []

def fetch_requests_for_collection(api_token, collection_id, start_date):
    """Fetch requests specific to one collection ID."""
    headers = {"Authorization": f"Bearer {api_token}"}
    url = "https://api.clearfeed.app/v1/rest/requests"
    
    requests_accumulated = []
    states = ["open", "pending", "solved", "closed"]
    
    for state in states:
        params = {
            "collection_id": collection_id,
            "after": start_date,
            "include": "messages",
            "limit": 50,
            "state": state
        }
        
        cursor = None
        while True:
            if cursor:
                params["next_cursor"] = cursor
            
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                batch = data.get("requests", [])
                if not batch: break
                
                requests_accumulated.extend(batch)
                
                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor: break
            except Exception as e:
                print(f"Error in batch: {e}")
                break
                
    return requests_accumulated