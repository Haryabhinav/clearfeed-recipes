import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from io import BytesIO

# --- UPDATED IMPORTS TO MATCH FILE STRUCTURE ---
from services import clearfeed_api, data_processor, intent_router, clustering_engine, prompt_generator
from utils import helpers

st.set_page_config(
    page_title="ClearFeed Insights",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

helpers.apply_custom_style()

# Session State Initialization
if "data_stage" not in st.session_state: st.session_state.data_stage = "init"
if "raw_data" not in st.session_state: st.session_state.raw_data = []
if "final_df" not in st.session_state: st.session_state.final_df = None
if "generated_definitions" not in st.session_state: st.session_state.generated_definitions = ""
if "validation_df" not in st.session_state: st.session_state.validation_df = None
if "collections_list" not in st.session_state: st.session_state.collections_list = []
if "ai_provider" not in st.session_state: st.session_state.ai_provider = "Gemini"

with st.sidebar:
    st.header("📊 Dashboard Config")
    st.markdown("---")
    with st.expander("🔐 API Credentials", expanded=True):
        cf_token = st.text_input("ClearFeed API Token", type="password")
        ai_provider = st.selectbox("Select AI Provider", ["Gemini", "OpenAI"])
        if ai_provider == "Gemini":
            ai_key = st.text_input("Google Gemini API Key", type="password")
            st.caption("Using model: gemini-2.5-flash")
        else:
            ai_key = st.text_input("OpenAI API Key", type="password")
            st.caption("Using model: gpt-4.1")
        
    if st.button("Link Account"):
        if not cf_token or not ai_key:
            st.warning("⚠️ Please provide both API keys.")
        else:
            validation_error = None
            colls = []
            with st.spinner(f"Validating {ai_provider} Credentials..."):
                colls = clearfeed_api.get_collections(cf_token)
                if not colls:
                    validation_error = "❌ Invalid ClearFeed Token."
                
                if not validation_error:
                    try:
                        if ai_provider == "Gemini":
                            import google.generativeai as genai
                            genai.configure(api_key=ai_key)
                            genai.GenerativeModel("gemini-2.5-flash").generate_content("Test")
                        else:
                            import openai
                            client = openai.OpenAI(api_key=ai_key)
                            client.models.list()
                    except Exception as e:
                        validation_error = f"❌ Invalid {ai_provider} Key/Quota.\nError: {str(e)}"

            if validation_error:
                st.error(validation_error)
            else:
                st.session_state.collections_list = colls
                st.session_state.ai_provider = ai_provider
                st.session_state.data_stage = "connected"
                st.success("✅ Connected Successfully!")
                time.sleep(1)
                st.rerun()

st.title("ClearFeed Intelligent Analyzer")
st.markdown("AI-Powered Support Ticket Analytics")

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

    if st.button("🚀 Run Analysis Pipeline", type="primary"):
        if not selected_names:
            st.error("Please select at least one collection.")
        else:
            status = st.status("Processing Data...", expanded=True)
            current_provider = st.session_state.get("ai_provider", "Gemini")

            try:
                start_date = date_range[0]
                end_date = date_range[1] if len(date_range) > 1 else datetime.now().date()
                start_date_iso = start_date.isoformat()
                end_date_iso = end_date.isoformat()

                cleaned_data = []

                progress_text = status.empty()
                progress_state = {"count": 0}

                def update_ui_progress(new_count):
                    progress_state["count"] += new_count
                    current = progress_state["count"]
                    progress_text.markdown(f"**📥 Fetching and Cleaning tickets... (Currently: {current} tickets)**")

                for name in selected_names:
                    cid = options[name]
                    tickets = clearfeed_api.fetch_requests_for_collection(
                        cf_token, cid, start_date_iso, end_date_iso, progress_callback=update_ui_progress
                    )
                    
                    valid_tickets = []
                    for t in tickets:
                        c_raw = t.get("created_at")
                        if c_raw:
                            try:
                                if isinstance(c_raw, str): t_d = datetime.strptime(c_raw[:10], "%Y-%m-%d").date()
                                else: t_d = datetime.fromtimestamp(c_raw / 1000).date()
                                if start_date <= t_d <= end_date: 
                                    valid_tickets.append(t)
                            except: valid_tickets.append(t)
                        else: valid_tickets.append(t)

                    if valid_tickets:
                        batch_clean = data_processor.process_raw_data(valid_tickets, collection_name=name)
                        cleaned_data.extend(batch_clean)
                
                status.write(f"✅ Processing Complete: {len(cleaned_data)} tickets prepared.")
                
                if not cleaned_data:
                    status.update(label="No Data Found", state="error")
                    st.error("No tickets found.")
                    st.stop()
                    
                status.write(f"🚦 Routing Intents ({current_provider})...")
                
                # --- UPDATED: Use intent_router ---
                buckets = intent_router.run_routing(cleaned_data, ai_key, provider=current_provider)
                
                status.write(f"🧠 Clustering Sub-topics ({current_provider})...")
                final_data = []
                
                n_feat, n_prob, n_howto = 0, 0, 0
                
                cats = [
                    ("feature_request", buckets.get('feature_request', []), n_feat),
                    ("problem_report", buckets.get('problem_report', []), n_prob),
                    ("how_to_question", buckets.get('how_to_question', []), n_howto)
                ]
                
                for name, data, n in cats:
                    if data:
                        if len(data) > 5:
                            # --- UPDATED: Use clustering_engine ---
                            res = clustering_engine.cluster_and_label_intent(name, data, n, ai_key, provider=current_provider)
                            final_data.extend(res)
                        else:
                             for item in data:
                                 item['cluster_category'] = f"General {name.replace('_', ' ').title()}"
                                 item['cluster_reasoning'] = "Too few items to cluster"
                             final_data.extend(data)
                
                if final_data:
                    df_temp = pd.DataFrame(final_data)
                    
                    df_temp['category_slug'] = df_temp['intent']
                    
                    def make_slug(text):
                        if not isinstance(text, str): return ""
                        slug = text.lower().strip()
                        slug = re.sub(r'[\s\-]+', '_', slug)
                        slug = re.sub(r'[^\w_]', '', slug)
                        return slug

                    df_temp['sub_category_slug'] = df_temp['cluster_category'].apply(make_slug)
                    
                    st.session_state.final_df = df_temp
                    
                    status.write("📝 Auto-generating Prompt Definitions...")
                    definitions_md = prompt_generator.generate_classification_prompt(
                        df_temp, ai_key, provider=current_provider
                    )
                    st.session_state.generated_definitions = definitions_md

                    status.write("🧪 Validating on 20 random tickets...")
                    val_df = prompt_generator.validate_prompt(
                        df_temp, definitions_md, ai_key, provider=current_provider
                    )
                    st.session_state.validation_df = val_df
                    
                    st.session_state.data_stage = "analyzed"
                    status.update(label="Pipeline Complete!", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="Analysis Failed", state="error")
                    st.error("No data produced.")
            
            except Exception as e:
                status.update(label="⚠️ Execution Stopped", state="error", expanded=True)
                st.error(f"An error occurred: {str(e)}")
            
            except BaseException as e:
                if type(e).__name__ != "RerunException":
                    status.update(label="🛑 Process Cancelled", state="error", expanded=False)
                    st.session_state.data_stage = "connected"
                    st.rerun()
                raise e

if st.session_state.data_stage == "analyzed" and st.session_state.final_df is not None:
    st.divider()
    st.subheader("📊 Analysis Results")
    df = st.session_state.final_df.copy()
    
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df.sort_values(by="created_at", ascending=False, inplace=True)
        df["created_at"] = df["created_at"].apply(lambda x: x.strftime('%b %d, %Y') if pd.notnull(x) else "")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickets", len(df))
    m2.metric("Feature Requests", len(df[df['intent'] == 'feature_request']))
    m3.metric("Problem Reports", len(df[df['intent'] == 'problem_report']))
    m4.metric("How-To Questions", len(df[df['intent'] == 'how_to_question']))
    
    st.divider()
    st.markdown("### 🧩 Deep Dive: Top Themes")
    
    c1, c2, c3 = st.columns(3)
    
    breakdown_cols = [
        ("✨ Feature Requests", "feature_request", c1),
        ("🐞 Problem Reports", "problem_report", c2),
        ("❓ How-To Questions", "how_to_question", c3)
    ]
    
    for title, intent_key, col in breakdown_cols:
        with col:
            st.info(f"**{title}**")
            subset = df[df['intent'] == intent_key]
            if not subset.empty:
                counts = subset['cluster_category'].value_counts().reset_index()
                counts.columns = ['Topic', 'Count']
                st.dataframe(counts, use_container_width=True, hide_index=True)
            else:
                st.caption("No tickets found.")
    
    st.divider()
    st.markdown("### 🧪 Validation & Inference Prompt")
    
    tab1, tab2 = st.tabs(["Production Prompt", "Accuracy Check (20 Samples)"])
    
    with tab1:
        st.success("✅ **Production-Ready Inference Prompt**")
        st.caption("Copy this entire prompt into your AI agent (ChatGPT, Zapier, LangChain) to classify new tickets automatically.")
        
        final_prompt_display = prompt_generator.build_inference_system_prompt(st.session_state.generated_definitions)
        st.text_area("Master Inference Prompt", value=final_prompt_display, height=400)
        
        st.download_button(
            label="📥 Download Prompt (.txt)",
            data=final_prompt_display,
            file_name="clearfeed_inference_prompt.txt",
            mime="text/plain"
        )

    with tab2:
        st.info("We tested the generated prompt above on 20 random tickets from your dataset. Here are the results:")
        if st.session_state.validation_df is not None:
            val_view = st.session_state.validation_df.copy()
            st.dataframe(
                val_view[['text', 'pred_category', 'pred_sub_category']], 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "text": st.column_config.TextColumn("Ticket Text", width="large"),
                    "pred_category": st.column_config.TextColumn("AI Category", width="medium"),
                    "pred_sub_category": st.column_config.TextColumn("AI Sub-topic", width="medium"),
                }
            )
        else:
            st.warning("Validation data missing.")

    st.divider()
    st.markdown("### 📋 Detailed Ticket Log")
    st.dataframe(
        df.drop(columns=["cluster_reasoning", "author_email"], errors='ignore'),
        use_container_width=True, hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
            "category_slug": st.column_config.TextColumn("Category Slug", help="Machine-readable category"),
            "sub_category_slug": st.column_config.TextColumn("Sub-Category Slug", help="Machine-readable sub-topic"),
        }
    )
    
    excel_data = helpers.convert_df_to_excel(df)
    st.download_button("📥 Download Report (Excel)", excel_data, "report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")