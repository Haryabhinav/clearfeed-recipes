import requests
import streamlit as st
import time
import logging
import concurrent.futures
from datetime import datetime, timedelta

# --- CONFIGURATION & LOGGING SETUP ---
API_BASE = "https://api.clearfeed.app/v1/rest"

# Configure Developer-Side Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600, show_spinner=False)
def get_collections(api_token):
    """Fetch all available collection IDs."""
    logger.info("📂 Fetching Collections list...")
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(f"{API_BASE}/collections", headers=headers)
        response.raise_for_status()
        collections = response.json().get("collections", [])
        logger.info(f"✅ Found {len(collections)} collections.")
        return collections
    except Exception as e:
        logger.error(f"❌ Error fetching collections: {e}")
        return []


def _ensure_iso_format(date_str, is_end_date=False):
    """Helper to ensure dates are strictly YYYY-MM-DDTHH:MM:SSZ"""
    if not date_str:
        return None
    
    clean_date = date_str.replace("Z", "")
    
    if "T" not in clean_date:
        if is_end_date:
            clean_date = f"{clean_date}T23:59:59"
        else:
            clean_date = f"{clean_date}T00:00:00"
            
    return f"{clean_date}Z"


def fetch_single_state(api_token, collection_id, state, api_start_date, filter_end_date):
    """
    Helper function to fetch requests for ONE specific state.
    Designed to be run inside a thread.
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_token}"})
    
    state_requests = []
    total_fetched_state = 0
    url = f"{API_BASE}/requests"
    
    params = {
        "collection_id": collection_id,
        "after": api_start_date, 
        "limit": 50,
        "include": "messages",
        "state": state
    }

    cursor = None
    while True:
        if cursor:
            params["next_cursor"] = cursor

        try:
            response = session.get(url, params=params, timeout=60)

            if response.status_code == 429:
                logger.warning(f"⚠️ Rate Limit Hit on state '{state}'. Sleeping 2s...")
                time.sleep(2)
                continue

            response.raise_for_status()
            data = response.json()

            batch = data.get("requests", [])
            if not batch:
                break

            # --- CLIENT-SIDE FILTERING ---
            if filter_end_date:
                filtered_batch = []
                for req in batch:
                    created_at = req.get("created_at")
                    if not created_at or created_at <= filter_end_date:
                        filtered_batch.append(req)
                batch = filtered_batch

            count = len(batch)
            state_requests.extend(batch)
            total_fetched_state += count
            
            # Optional: Log per batch if needed, but can get noisy in threads
            # logger.info(f"   ↳ {state}: +{count}")

            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        except Exception as e:
            logger.error(f"❌ Error on state '{state}': {e}")
            break
            
    session.close()
    return state_requests


def fetch_requests_for_collection(api_token, collection_id, start_date, end_date=None, progress_callback=None):
    """
    Fetch requests using ThreadPoolExecutor to run 4 states in parallel.
    """
    # Prepare Dates
    api_start_date = _ensure_iso_format(start_date, is_end_date=False)
    filter_end_date = _ensure_iso_format(end_date, is_end_date=True)

    all_requests = []
    states = ["open", "pending", "solved", "closed"]
    
    logger.info(f"🚀 Starting threaded fetch for collection {collection_id} (4 states parallel)")

    # --- THREADPOOL EXECUTOR ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all 4 state tasks at once
        future_to_state = {
            executor.submit(
                fetch_single_state, 
                api_token, 
                collection_id, 
                state, 
                api_start_date, 
                filter_end_date
            ): state for state in states
        }
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_state):
            state = future_to_state[future]
            try:
                data = future.result()
                all_requests.extend(data)
                
                count = len(data)
                logger.info(f"   ✅ Finished state '{state}': Retrieved {count} tickets")
                
                # Update progress bar if provided (Note: Streamlit is not always thread-safe with UI updates from threads)
                if progress_callback:
                    progress_callback(count)
                    
            except Exception as exc:
                logger.error(f"❌ State '{state}' generated an exception: {exc}")

    logger.info(f"🟢 Finished fetching collection {collection_id}. Total: {len(all_requests)}")
    return all_requests


def run_optimized_extraction(api_token, start_date=None, end_date=None):
    """
    Main extraction runner.
    """
    start_time = time.time()

    # Defaults
    if not start_date:
        start_date = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    start_date = _ensure_iso_format(start_date)
    end_date = _ensure_iso_format(end_date, is_end_date=True)

    logger.info(f"📅 Extracting data from: {start_date} to {end_date}")

    collections = get_collections(api_token)
    if not collections:
        return []

    all_tickets = []
    global_counter = 0

    for collection in collections:
        collection_id = collection['id']
        collection_name = collection.get('name', 'Unknown')

        logger.info(f"🔵 [{collection_name}] Starting fetch...")

        try:
            requests_data = fetch_requests_for_collection(
                api_token,
                collection_id,
                start_date,
                end_date
            )
            count = len(requests_data)

            all_tickets.extend(requests_data)
            global_counter += count

            logger.info(f"📈 GLOBAL UPDATE: {global_counter} tickets fetched so far (Finished: {collection_name})")

        except Exception as e:
            logger.error(f"❌ Failed to fetch {collection_name}: {e}")

    duration = time.time() - start_time
    logger.info(f"🏁 DONE in {duration:.2f}s. Total Requests: {len(all_tickets)}")
    return all_tickets

# ----------------------------------------------------
# TO RUN THIS IN STREAMLIT:
# if st.button("Run Extraction"):
#     data = run_optimized_extraction(YOUR_API_TOKEN, "2023-10-01", "2024-01-01")
#     st.write(f"Fetched {len(data)} tickets.")
# ----------------------------------------------------