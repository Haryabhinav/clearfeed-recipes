import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

# Custom Modules: Handles API calls, data cleaning, and AI processing
from services import clearfeed_api, data_processor, ai_engine
from utils import helpers

# --- PAGE CONFIGURATION ---
# Sets up the browser tab title, icon, and layout width
st.set_page_config(
    page_title="ClearFeed Insights",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injects custom CSS for better UI styling
helpers.apply_custom_style()

# --- STATE MANAGEMENT ---
# Initialize session state variables to persist data across re-runs
if "data_stage" not in st.session_state:
    st.session_state.data_stage = "init"
if "raw_data" not in st.session_state:
    st.session_state.raw_data = []
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "collections_list" not in st.session_state:
    st.session_state.collections_list = []
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "Gemini"

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("📊 Dashboard Config")
    st.markdown("---")
    
    # Credentials Input Section
    with st.expander("🔐 API Credentials", expanded=True):
        cf_token = st.text_input("ClearFeed API Token", type="password")
        ai_provider = st.selectbox("Select AI Provider", ["Gemini", "OpenAI"])
        
        # Conditional input based on selected provider
        if ai_provider == "Gemini":
            ai_key = st.text_input("Google Gemini API Key", type="password")
            st.caption("Using model: gemini-2.5-flash")
        else:
            ai_key = st.text_input("OpenAI API Key", type="password")
            st.caption("Using model: gpt-4o")
        
    # Link Account Button: Validates keys before proceeding
    if st.button("Link Account"):
        if not cf_token or not ai_key:
            st.warning("⚠️ Please provide both API keys.")
        else:
            with st.spinner(f"Validating {ai_provider} Credentials..."):
                # 1. Validate ClearFeed Token
                colls = clearfeed_api.get_collections(cf_token)
                if not colls:
                    st.error("❌ Invalid ClearFeed Token.")
                    st.stop()
                
                # 2. Validate AI Provider Key
                ai_valid = False
                try:
                    if ai_provider == "Gemini":
                        import google.generativeai as genai
                        genai.configure(api_key=ai_key)
                        genai.GenerativeModel("gemini-2.5-flash").generate_content("Test")
                    else:
                        import openai
                        client = openai.OpenAI(api_key=ai_key)
                        client.models.list()
                    ai_valid = True
                except Exception as e:
                    st.error(f"❌ Invalid {ai_provider} Key. Error: {str(e)}")
                    st.stop()

                # If successful, update state and reload
                if colls and ai_valid:
                    st.session_state.collections_list = colls
                    st.session_state.ai_provider = ai_provider
                    st.session_state.data_stage = "connected"
                    st.success("✅ Connected Successfully!")
                    time.sleep(1)
                    st.rerun()

# --- MAIN UI ---
st.title("ClearFeed Intelligent Analyzer")
st.markdown("AI-Powered Support Ticket Analytics")

# Display initial instruction if keys aren't linked
if st.session_state.data_stage == "init":
    st.info("👈 Please connect your API keys in the sidebar to begin.")

# Main Analysis Workflow (Active after connection)
elif st.session_state.data_stage in ["connected", "extracted", "analyzed"]:
    
    # 1. Data Ingestion Configuration
    st.subheader("1. Data Ingestion")
    col1, col2 = st.columns(2)
    with col1:
        # Date picker with default range of last 90 days
        default_start = datetime.now() - timedelta(days=90)
        date_range = st.date_input("Select Date Range", value=(default_start, datetime.now()))
    
    with col2:
        # Multiselect for specific collections (Channels)
        options = {c['name']: c['id'] for c in st.session_state.collections_list}
        selected_names = st.multiselect("Select Collections", options=list(options.keys()), default=list(options.keys()))
    
    # 2. Advanced Clustering Settings (User defines sub-topic granularity)
    with st.expander("⚙️ Advanced Cluster Settings"):
        st.caption("Set number of sub-topics for each category")
        c1, c2, c3 = st.columns(3)
        n_feat = c1.number_input("Features", value=5, min_value=2)
        n_prob = c2.number_input("Problems", value=5, min_value=2)
        n_howto = c3.number_input("How-To", value=5, min_value=2)

    # 3. Execution Pipeline
    if st.button("🚀 Run Analysis Pipeline", type="primary"):
        if not selected_names:
            st.error("Please select at least one collection.")
        else:
            # Initialize Status Container
            status = st.status("Processing Data...", expanded=True)
            current_provider = st.session_state.get("ai_provider", "Gemini")

            # --- Step 1: Set Date Boundaries ---
            start_date = date_range[0]
            end_date = date_range[1] if len(date_range) > 1 else datetime.now().date()
            
            # --- Step 2: Fetch & Filter Data ---
            status.write("📥 Fetching tickets from ClearFeed...")
            raw_bucket = []
            start_date_iso = start_date.isoformat()
            
            for name in selected_names:
                cid = options[name]
                # API Call: Gets tickets created AFTER start_date
                tickets = clearfeed_api.fetch_requests_for_collection(cf_token, cid, start_date_iso)
                
                # Manual Filter: Removes tickets created AFTER end_date
                valid_tickets = []
                for t in tickets:
                    created_at_raw = t.get("created_at")
                    
                    if created_at_raw:
                        try:
                            # Normalize date format (ISO or Timestamp)
                            if isinstance(created_at_raw, str):
                                t_date_str = created_at_raw[:10] 
                                t_date = datetime.strptime(t_date_str, "%Y-%m-%d").date()
                            elif isinstance(created_at_raw, (int, float)):
                                t_date = datetime.fromtimestamp(created_at_raw / 1000).date()
                            
                            # Filter Logic: Start <= Ticket Date <= End
                            if start_date <= t_date <= end_date:
                                valid_tickets.append(t)
                        except:
                            valid_tickets.append(t) # Fallback: keep if parsing fails
                
                raw_bucket.extend(valid_tickets)
            
            status.write(f"✅ Filtered to {len(raw_bucket)} tickets within date range.")

            if not raw_bucket:
                status.update(label="No Data Found", state="error")
                st.error(f"No tickets found between {start_date} and {end_date}.")
                st.stop()
                
            # --- Step 3: Cleaning & Formatting ---
            status.write("🧹 Cleaning Transcripts & preprocessing...")
            cleaned_data = data_processor.process_raw_data(raw_bucket)
            
            # --- Step 4: AI Intent Routing (Feature vs Problem vs How-To) ---
            status.write(f"🚦 Routing Intents ({current_provider})...")
            buckets = ai_engine.run_routing(cleaned_data, ai_key, provider=current_provider)
            
            # --- Step 5: AI Clustering (Sub-topic generation) ---
            status.write(f"🧠 Clustering Sub-topics ({current_provider})...")
            final_data = []
            
            # Define clusters to process
            categories = [
                ("feature_request", buckets['feature_request'], n_feat),
                ("problem_report", buckets['problem_report'], n_prob),
                ("how_to_question", buckets.get('how_to_question', []), n_howto)
            ]

            for name, data, n_clusters in categories:
                if data:
                    res = ai_engine.cluster_and_label_intent(name, data, n_clusters, ai_key, provider=current_provider)
                    final_data.extend(res)
            
            # Finalize State
            if final_data:
                st.session_state.final_df = pd.DataFrame(final_data)
                st.session_state.data_stage = "analyzed"
                status.update(label="Pipeline Complete!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Analysis Failed", state="error")
                st.error("Pipeline ran but produced no final data.")

# --- RESULTS VIEW ---
# Displays metrics and tables after successful analysis
if st.session_state.data_stage == "analyzed" and st.session_state.final_df is not None:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Prepare Data for Display
    df = st.session_state.final_df.copy()
    
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df.sort_values(by="created_at", ascending=False, inplace=True)
    
    display_df = df.copy()
    cols_to_drop = ["cluster_reasoning", "author_email"] 
    display_df = display_df.drop(columns=[c for c in cols_to_drop if c in display_df.columns], errors='ignore')

    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].apply(lambda x: x.strftime('%b %d, %Y') if pd.notnull(x) else "")

    # KPI Metrics Section
    m1, m2, m3, m4 = st.columns(4)
    
    feat_c = len(df[df['intent'] == 'feature_request'])
    prob_c = len(df[df['intent'] == 'problem_report'])
    how_c = len(df[df['intent'] == 'how_to_question'])
    
    m1.metric("Total Tickets", len(df))
    m2.metric("Feature Requests", feat_c)
    m3.metric("Problem Reports", prob_c)
    m4.metric("How-To Questions", how_c)
    
    # Detailed Data Table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "request_id": st.column_config.TextColumn("System ID", width="small"),
            "intent": st.column_config.TextColumn("Intent", width="small"),
            "explanation": st.column_config.TextColumn("AI Reasoning", width="medium"),
            "cluster_category": st.column_config.TextColumn("Sub-Topic", width="large"),
            "text": st.column_config.TextColumn("Conversation Transcript", width="large"),
            "channel": st.column_config.TextColumn("Channel", width="medium"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "author_name": st.column_config.TextColumn("Author", width="medium"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
            "state": st.column_config.TextColumn("Status", width="small"),
            "created_at": st.column_config.TextColumn("Date", width="small"),
        }
    )
    
    # Excel Export Button
    excel_data = helpers.convert_df_to_excel(df)
    st.download_button(
        label="📥 Download Full Report (.xlsx)",
        data=excel_data,
        file_name=f"clearfeed_analysis_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )