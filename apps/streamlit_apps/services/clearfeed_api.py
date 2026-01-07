# File: services/clearfeed_api.py

import requests
import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- 1. GET COLLECTIONS FUNCTION ---
@st.cache_data(ttl=600, show_spinner=False)
def get_collections(api_token):
    """
    Fetch all available collection IDs.
    """
    # Note: Use .app domain for this endpoint
    url = "https://api.clearfeed.app/v1/rest/collections"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 401:
            st.error("🚨 Authentication Failed: Invalid API Token.")
            return []
        if response.status_code != 200:
            st.error(f"Error fetching collections (Status {response.status_code}): {response.text}")
            return []
            
        data = response.json()
        return data.get("collections", [])
        
    except Exception as e:
        st.error(f"🚨 Network Error fetching collections: {e}")
        return []

# --- 2. FETCH REQUESTS FUNCTION ---
def fetch_requests_for_collection(api_token, collection_id, start_date, end_date=None, progress_callback=None):
    """
    Fetch requests using Parallel Workers with Retry Logic.
    """
    headers = {"Authorization": f"Bearer {api_token}"}
    url = "https://api.clearfeed.app/v1/rest/requests"
    
    states = ["open", "pending", "solved", "closed"]
    all_tickets = []
    
    # Pre-calculate end timestamp
    end_ts = None
    if end_date:
        try:
            if isinstance(end_date, str):
                dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                end_ts = dt.timestamp() * 1000
        except: 
            pass

    def parse_timestamp(date_val):
        if not date_val: return 0
        if isinstance(date_val, (int, float)): return date_val
        if isinstance(date_val, str):
            try:
                dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                return dt.timestamp() * 1000
            except ValueError:
                return 0
        return 0

    def fetch_state_data(state):
        local_tickets = []
        params = {
            "collection_id": collection_id,
            "after": start_date,
            "include": "messages",
            "limit": 100, 
            "state": state
        }
        
        if end_date: params["before"] = end_date
            
        cursor = None
        
        while True:
            if cursor: params["next_cursor"] = cursor
            
            # Retry Logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    if response.status_code == 429:
                        time.sleep(2 * (attempt + 1))
                        continue
                    response.raise_for_status()
                    data = response.json()
                    break 
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"FAILED fetch state '{state}': {e}")
                        return local_tickets
                    time.sleep(1)
            
            batch = data.get("requests", [])
            if not batch: break
            
            # Timestamp check
            if end_ts and batch:
                first_val = batch[0].get('created_at')
                first_ts = parse_timestamp(first_val)
                if first_ts > 0 and first_ts > (end_ts + 86400000): 
                    break
            
            local_tickets.extend(batch)
            
            if progress_callback:
                try: progress_callback(len(batch))
                except: pass

            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor: break
            time.sleep(0.1) 
        
        return local_tickets

    # Parallel Execution
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for state in states:
            future = executor.submit(fetch_state_data, state)
            add_script_run_ctx(future) # Context injection
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                res = future.result()
                all_tickets.extend(res)
            except Exception as e:
                print(f"Worker crashed: {e}")

    return all_tickets