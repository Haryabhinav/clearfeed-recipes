import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

# Custom Modules
from services import clearfeed_api, data_processor, ai_engine
from utils import helpers

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ClearFeed Insights Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

helpers.apply_custom_style()

# --- STATE ---
if "data_stage" not in st.session_state:
    st.session_state.data_stage = "init"
if "raw_data" not in st.session_state:
    st.session_state.raw_data = []
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "collections_list" not in st.session_state:
    st.session_state.collections_list = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔌 Configuration")
    with st.expander("API Credentials", expanded=True):
        cf_token = st.text_input("ClearFeed API Token", type="password")
        ai_key = st.text_input("Gemini API Key", type="password")
        
    if st.button("🔗 Connect & Verify"):
        if not cf_token or not ai_key:
            st.warning("Please provide both API keys.")
        else:
            with st.spinner("Verifying connection..."):
                colls = clearfeed_api.get_collections(cf_token)
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=ai_key)
                    m = genai.GenerativeModel("gemini-2.5-flash-lite")
                    m.generate_content("Hi")
                    ai_valid = True
                except:
                    ai_valid = False

                if colls and ai_valid:
                    st.session_state.collections_list = colls
                    st.session_state.data_stage = "connected"
                    st.success("✅ Systems Online")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Connection Failed.")

# --- MAIN UI ---
st.title("🤖 ClearFeed Intelligent Analyzer")
st.markdown("Industry-grade RAG & Clustering pipeline for support ticket analysis.")

if st.session_state.data_stage == "init":
    st.info("👈 Please connect your API keys in the sidebar to begin.")

elif st.session_state.data_stage in ["connected", "extracted", "analyzed"]:
    
    st.subheader("1. Data Ingestion")
    col1, col2 = st.columns(2)
    with col1:
        default_start = datetime.now() - timedelta(days=90)
        date_range = st.date_input("Select Date Range", value=(default_start, datetime.now()))
    
    with col2:
        options = {c['name']: c['id'] for c in st.session_state.collections_list}
        selected_names = st.multiselect("Select Collections", options=list(options.keys()), default=list(options.keys()))
    
    with st.expander("⚙️ Advanced Cluster Settings"):
        c1, c2, c3 = st.columns(3)
        n_howto = c1.number_input("How-To Clusters", value=5, min_value=2)
        n_problem = c2.number_input("Problem Clusters", value=5, min_value=2)
        n_request = c3.number_input("Request Clusters", value=5, min_value=2)

    if st.button("🚀 Run Analysis Pipeline", type="primary"):
        if not selected_names:
            st.error("Please select at least one collection.")
        else:
            status = st.status("Processing Data...", expanded=True)
            
            # 1. Extraction
            status.write("📥 Fetching tickets from ClearFeed...")
            raw_bucket = []
            start_date = date_range[0] if isinstance(date_range, tuple) else date_range
            start_date_iso = start_date.isoformat()
            
            for name in selected_names:
                cid = options[name]
                tickets = clearfeed_api.fetch_requests_for_collection(cf_token, cid, start_date_iso)
                raw_bucket.extend(tickets)
            
            if not raw_bucket:
                status.update(label="No Data Found", state="error")
                st.error("No tickets found in this date range.")
                st.stop()
                
            # 2. Cleaning
            status.write("🧹 Cleaning Transcripts & preprocessing...")
            cleaned_data = data_processor.process_raw_data(raw_bucket)
            
            # 3. Routing
            status.write("🚦 Routing Intents (Gemini)...")
            buckets = ai_engine.run_routing(cleaned_data, ai_key)
            
            # 4. Clustering
            status.write("🧠 Clustering & Labeling Sub-topics...")
            final_data = []
            
            if buckets['how_to']:
                res = ai_engine.cluster_and_label_intent("how_to", buckets['how_to'], n_howto, ai_key)
                final_data.extend(res)
            
            if buckets['problem_report']:
                res = ai_engine.cluster_and_label_intent("problem_report", buckets['problem_report'], n_problem, ai_key)
                final_data.extend(res)

            if buckets['request']:
                res = ai_engine.cluster_and_label_intent("request", buckets['request'], n_request, ai_key)
                final_data.extend(res)
                
            if final_data:
                st.session_state.final_df = pd.DataFrame(final_data)
                st.session_state.data_stage = "analyzed"
                status.update(label="Pipeline Complete!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Analysis Failed", state="error")
                st.error("Pipeline ran but produced no final data.")

# --- RESULTS VIEW ---
if st.session_state.data_stage == "analyzed" and st.session_state.final_df is not None:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    df = st.session_state.final_df
    
    # SORTING: Sort by Date Descending
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df.sort_values(by="created_at", ascending=False, inplace=True)
    
    # Display Copy
    display_df = df.copy()
    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].apply(lambda x: x.strftime('%b %d, %Y') if pd.notnull(x) else "")

    # METRICS (Updated to 4 columns)
    m1, m2, m3, m4 = st.columns(4)
    
    how_to_cnt = len(df[df['intent'] == 'how_to']) if 'intent' in df.columns else 0
    prob_cnt = len(df[df['intent'] == 'problem_report']) if 'intent' in df.columns else 0
    req_cnt = len(df[df['intent'] == 'request']) if 'intent' in df.columns else 0
    
    m1.metric("Total Tickets", len(df))
    m2.metric("How-To Questions", how_to_cnt)  # <-- Added
    m3.metric("Problem Reports", prob_cnt)
    m4.metric("Feature Requests", req_cnt)
    
    # Main Dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "cf_id": st.column_config.TextColumn("Ticket ID", width="small"),
            "intent": st.column_config.TextColumn("Intent", width="small"),
            "cluster_category": st.column_config.TextColumn("Sub-Topic", width="medium"),
            "text": st.column_config.TextColumn("Conversation Transcript", width="large"),
            "channel": st.column_config.TextColumn("Channel", width="medium"),
            "author_name": st.column_config.TextColumn("Author", width="medium"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
            "state": st.column_config.TextColumn("Status", width="small"),
            "created_at": st.column_config.TextColumn("Date", width="small"),
            
            # Hidden
            "request_id": None,
            "cluster_reasoning": None,
            "source": None,
            "author_email": None
        }
    )
    
    excel_data = helpers.convert_df_to_excel(display_df)
    st.download_button(
        label="📥 Download Report (.xlsx)",
        data=excel_data,
        file_name=f"clearfeed_analysis_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )