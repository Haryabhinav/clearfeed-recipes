import json
import time
import re
import numpy as np
from sklearn.cluster import KMeans
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # <--- NEW IMPORT
import openai
import streamlit as st

# --- HELPER: CLEAN AI JSON ---
def clean_json_response(text):
    """
    Removes markdown code blocks (```json ... ```) to ensure valid JSON parsing.
    """
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# --- OPENAI SPECIFIC HELPERS ---
def call_openai_json(api_key, prompt, model="gpt-4o-mini"):
    """
    Calls OpenAI Chat Completion.
    STOPS EXECUTION IMMEDIATELY if Quota is exceeded.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    except openai.RateLimitError:
        st.error("🚨 OpenAI Quota Exceeded! Execution stopped. Please add funds at platform.openai.com/billing.")
        st.stop() 
        return "{}"
    except openai.AuthenticationError:
        st.error("🚨 Invalid OpenAI API Key. Please check your credentials.")
        st.stop()
        return "{}"
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return "{}"

def get_openai_embeddings(texts, api_key):
    """
    Generates embeddings.
    STOPS EXECUTION IMMEDIATELY if Quota is exceeded.
    """
    client = openai.OpenAI(api_key=api_key)
    embeddings = []
    
    # OpenAI handles larger batches, but we stick to 50 for safety
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        try:
            # Replace newlines for better embedding quality
            clean_batch = [t.replace("\n", " ") for t in batch]
            resp = client.embeddings.create(input=clean_batch, model="text-embedding-3-small")
            embeddings.extend([d.embedding for d in resp.data])

        except openai.RateLimitError:
            st.error("🚨 OpenAI Quota Exceeded during embedding! Execution stopped.")
            st.stop()
            return np.array([])
        except Exception as e:
            st.error(f"OpenAI Embedding Error: {str(e)}")
            # Fallback: Zero vectors (Dimension 1536 for small model)
            embeddings.extend([[0.0]*1536 for _ in batch])
            
    return np.array(embeddings)

# --- GEMINI SPECIFIC HELPERS (UPDATED) ---
def get_gemini_embeddings(texts, api_key):
    """
    Generates embeddings using Google's 'text-embedding-004'.
    STOPS EXECUTION IMMEDIATELY if Quota is exceeded.
    """
    genai.configure(api_key=api_key)
    embeddings = []
    
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="clustering"
            )
            embeddings.extend(result['embedding'])
            time.sleep(0.5) # Rate limit protection

        except google_exceptions.ResourceExhausted:
            st.error("🚨 Gemini Quota Exceeded! Execution stopped. Please check your Google Cloud quota.")
            st.stop()
            return np.array([])
        except Exception as e:
            # General fallback for other errors
            embeddings.extend([[0.0]*768 for _ in batch])
            
    return np.array(embeddings)

# --- UNIFIED ROUTING ---
def classify_batch(requests_batch, api_key, provider):
    batch_text = ""
    for req in requests_batch:
        # Context: First 500 chars
        clean_text = req['text'][:500].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    # PROMPT: 3 Intents containing all definitions
    prompt = f"""
    You are a Support Ticket Classifier.
    Task: Label the following batch of tickets into exactly one of these 3 categories.

    ### LABELS & DEFINITIONS:
    1. **feature_request**: 
       - The customer is requesting a new feature, enhancement, OR general assistance (e.g., account changes, new integrations).
       - INCLUDES: Any "Request" that is NOT a bug report.
    
    2. **problem_report**: 
       - The customer reports an issue, error, bug, or unexpected behavior.
       - INCLUDES: General problems with product features or configuration.
    
    3. **how_to_question**: 
       - The customer asks for clarification or instruction on how to use a feature or product.

    ### INPUT BATCH DATA:
    {batch_text}

    ### OUTPUT INSTRUCTIONS:
    1. Return strictly valid JSON.
    2. Assign the SINGLE most relevant label.
    
    Format:
    {{
        "ID_123": {{ "label": "problem_report", "explanation": "Agent confirmed logic error." }},
        "ID_456": {{ "label": "feature_request", "explanation": "User requested a new separate account." }}
    }}
    """
    
    if provider == "OpenAI":
        json_str = call_openai_json(api_key, prompt)
        return json.loads(json_str)
    else:
        # GEMINI BRANCH (Updated with Quota Handling)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        try:
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(clean_json_response(response.text))
            
        except google_exceptions.ResourceExhausted:
            st.error("🚨 Gemini Quota Exceeded! Execution stopped.")
            st.stop()
            return {}
        except Exception as e:
            st.error(f"Gemini Routing Failed: {e}")
            return {}

def run_routing(cleaned_data, api_key, provider="Gemini"):
    """
    Main routing function with ROBUST ID matching.
    """
    # STRICT 3 BUCKETS
    buckets = {
        "feature_request": [],
        "problem_report": [],
        "how_to_question": []
    }
    
    batch_size = 15
    progress_bar = st.progress(0)
    total = len(cleaned_data)
    
    for i in range(0, total, batch_size):
        batch = cleaned_data[i : i + batch_size]
        if not batch: continue
        
        # 1. Get AI Response
        results_map = classify_batch(batch, api_key, provider)
        
        # 2. NORMALIZE KEYS (The Fix for "Auto-routed")
        normalized_results = {}
        if isinstance(results_map, dict):
            for k, v in results_map.items():
                clean_key = re.sub(r"\D", "", str(k)) # Keep only digits
                normalized_results[clean_key] = v
        elif isinstance(results_map, list):
            for item in results_map:
                if isinstance(item, dict):
                    for k, v in item.items():
                        clean_key = re.sub(r"\D", "", str(k))
                        normalized_results[clean_key] = v

        # 3. Match Tickets
        for req in batch:
            req_id_str = str(req['request_id'])
            res = normalized_results.get(req_id_str, {})
            
            raw_label = res.get("label", "problem_report") 
            if isinstance(raw_label, list): raw_label = raw_label[0]

            # SAFETY MAPPING
            if raw_label == "bug": raw_label = "problem_report"
            if raw_label == "request": raw_label = "feature_request"

            if raw_label not in buckets: 
                raw_label = "problem_report"
            
            # Explanation Logic
            if not res:
                req['explanation'] = "Auto-routed (AI response missing for this ID)"
            else:
                req['explanation'] = res.get("explanation", "AI Classified")

            req['intent'] = raw_label
            buckets[raw_label].append(req)
            
        progress_bar.progress(min((i + batch_size) / total, 1.0))
        if provider == "Gemini": time.sleep(1) 
        
    return buckets

# --- UNIFIED CLUSTERING ---
def cluster_and_label_intent(intent_name, data_list, num_clusters, api_key, provider="Gemini"):
    """
    Clusters data and generates labels.
    INCLUDES FIX FOR 'UNLABELED' GROUPS & QUOTA STOPPING.
    """
    if len(data_list) < num_clusters:
        for item in data_list:
            item['cluster_category'] = f"General {intent_name.replace('_', ' ').title()}"
            item['cluster_reasoning'] = "Not enough data to cluster"
        return data_list
        
    texts = [x['text'] for x in data_list]
    
    # 1. Embeddings
    if provider == "OpenAI":
        embeddings = get_openai_embeddings(texts, api_key)
    else:
        embeddings = get_gemini_embeddings(texts, api_key)
    
    if len(embeddings) == 0:
        for item in data_list:
            item['cluster_category'] = "Processing Failed"
            item['cluster_reasoning'] = "Embeddings could not be generated"
        return data_list

    # 2. KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    
    # 3. Sampling
    samples_map = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        sample_idxs = np.random.choice(indices, min(len(indices), 5), replace=False)
        samples_map[f"{i}"] = [texts[idx] for idx in sample_idxs]
        
    # 4. Labeling Prompt
    prompt = f"""
    You are a Senior Support Analyst. 
    Group Topic: '{intent_name}'.
    
    TASK:
    1. Read the sample tickets in each group.
    2. Create a short Title (3-5 words) that describes the specific sub-topic.
    3. Provide a short reasoning.

    INPUT GROUPS:
    """
    for k, v in samples_map.items():
        clean_samples = [t[:300].replace("\n", " ") for t in v]
        prompt += f"\nGroup {k}: {json.dumps(clean_samples)}"

    prompt += """
    OUTPUT JSON (Strictly use ONLY the Group Number "0", "1"... as key):
    {
        "0": { "category": "Login Issues", "reasoning": "Users failing to sign in." }
    }
    """
    
    labels_map = {}
    try:
        response_text = ""
        if provider == "OpenAI":
            response_text = call_openai_json(api_key, prompt)
        else:
            # GEMINI BRANCH (Updated)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            response_text = clean_json_response(resp.text)
            
        labels_map = json.loads(response_text)

    except google_exceptions.ResourceExhausted:
        st.error("🚨 Gemini Quota Exceeded during Labeling! Execution stopped.")
        st.stop()
        labels_map = {}
    except Exception as e:
        labels_map = {}

    if isinstance(labels_map, list):
        temp = {}
        for item in labels_map:
            if isinstance(item, dict): temp.update(item)
        labels_map = temp

    # 5. Apply Labels
    normalized_labels = {}
    for k, v in labels_map.items():
        clean_key = re.sub(r"\D", "", str(k))
        normalized_labels[clean_key] = v

    for idx, cluster_id in enumerate(labels):
        c_id_str = str(cluster_id)
        info = normalized_labels.get(c_id_str)
        
        if info:
            cat = info.get('category', f"Cluster {cluster_id}")
            reason = info.get('reasoning', "AI labeled")
        else:
            cat = f"Unlabeled {intent_name.title()} Group {cluster_id}"
            reason = "AI Labeling step failed to match key"
        
        data_list[idx]['cluster_category'] = cat
        data_list[idx]['cluster_reasoning'] = reason
        
    return data_list