import json
import time
import re
import numpy as np
from sklearn.cluster import KMeans
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import openai
import streamlit as st

# --- HELPER: CLEAN AI JSON ---
def clean_json_response(text):
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# --- OPENAI SPECIFIC HELPERS ---
def call_openai_json(api_key, prompt, model="gpt-4.1"):
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
        raise Exception("🚨 OpenAI Quota Exceeded! Execution stopped. Check billing.")
    except openai.AuthenticationError:
        raise Exception("🚨 Invalid OpenAI API Key.")
    except Exception:
        return "{}"

# --- GEMINI EMBEDDINGS (With Retry) ---
def get_gemini_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    embeddings = []
    
    # Process in small batches to avoid timeouts
    for i in range(0, len(texts), 20):
        batch = texts[i:i+20]
        try:
            # Using the embedding model specifically
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=[t[:500] for t in batch], # Truncate for safety
                task_type="clustering"
            )
            embeddings.extend(result['embedding'])
            time.sleep(0.5) # Rate limit safety

        except Exception as e:
            # If embedding fails, return empty to trigger fallback logic
            print(f"Embedding failed: {e}")
            return [], 0.0
            
    return np.array(embeddings), 1.0

# --- OPENAI EMBEDDINGS ---
def get_openai_embeddings(texts, api_key):
    client = openai.OpenAI(api_key=api_key)
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        try:
            clean_batch = [t.replace("\n", " ") for t in batch]
            resp = client.embeddings.create(input=clean_batch, model="text-embedding-3-small")
            embeddings.extend([d.embedding for d in resp.data])
        except Exception:
             # Fail fast
            return [], 0.0
    return np.array(embeddings), 1.0

# --- 1. CLASSIFICATION (ROUTING) ---
def classify_batch(requests_batch, api_key, provider):
    batch_text = ""
    for req in requests_batch:
        # Strict context limit (first 200 chars are usually enough for intent)
        clean_text = req['text'][:200].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    # 🟢 OPTIMIZED PROMPT: Asking for simple Key-Value pairs only
    prompt = f"""
    You are a Support Ticket Classifier.
    Task: Label each ticket with exactly ONE category.
    
    Categories:
    - feature_request
    - problem_report
    - how_to_question
    
    Input Data:
    {batch_text}
    
    OUTPUT JSON FORMAT (Strictly Key-Value):
    {{
      "ID_123": "feature_request",
      "ID_456": "problem_report"
    }}
    """
    
    if provider == "OpenAI":
        return json.loads(call_openai_json(api_key, prompt, model="gpt-4.1"))
    else:
        genai.configure(api_key=api_key)
        # Use a stable model
        model = genai.GenerativeModel("gemini-2.5-flash-lite") 
        try:
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(clean_json_response(response.text))
        except Exception as e:
            print(f"🚨 Batch Classification Failed: {e}")
            return {}

def run_routing(cleaned_data, api_key, provider="Gemini"):
    buckets = {
        "feature_request": [],
        "problem_report": [],
        "how_to_question": [],
        "unclassified": [] # New bucket for failures
    }
    
    batch_size = 50 # Smaller batch size for higher accuracy
    progress_bar = st.progress(0)
    total = len(cleaned_data)
    
    # 🟢 FIXED: Restored missing 'for' loop and removed incorrect 'model=' line
    for i in range(0, total, batch_size):
        batch = cleaned_data[i : i + batch_size]
        if not batch: continue
        
        results_map = classify_batch(batch, api_key, provider)
        
        # Normalize Keys (Handle ID_123 vs 123)
        normalized_results = {}
        if results_map:
            for k, v in results_map.items():
                clean_key = re.sub(r"\D", "", str(k))
                normalized_results[clean_key] = v

        for req in batch:
            req_id_str = str(req['request_id'])
            res = normalized_results.get(req_id_str)
            
            # Handle String vs Dict vs None
            raw_label = "unclassified"
            explanation = "AI could not classify"
            
            if isinstance(res, str):
                # AI returned simple format: {"ID": "label"} -> CORRECT
                raw_label = res
                explanation = "AI Classified"
            elif isinstance(res, dict):
                # AI returned complex format: {"ID": {"label": "..."}}
                raw_label = res.get("label", "unclassified")
                explanation = res.get("explanation", "AI Classified")
            
            # Normalize label
            raw_label = raw_label.lower().strip()
            if "feature" in raw_label: raw_label = "feature_request"
            elif "problem" in raw_label or "bug" in raw_label: raw_label = "problem_report"
            elif "how" in raw_label or "question" in raw_label: raw_label = "how_to_question"
            
            # Assign to bucket
            if raw_label in buckets:
                req['intent'] = raw_label
                req['explanation'] = explanation
                buckets[raw_label].append(req)
            else:
                # If AI returned garbage, put in unclassified (don't fake it as problem_report)
                req['intent'] = "unclassified"
                req['explanation'] = f"Raw AI output: {raw_label}"
                buckets["unclassified"].append(req)
            
        progress_bar.progress(min((i + batch_size) / total, 1.0))
        time.sleep(0.5) 
        
    return buckets

# --- 2. CLUSTERING FUNCTION ---
def cluster_and_label_intent(intent_name, data_list, num_clusters, api_key, provider="Gemini"):
    if len(data_list) < num_clusters:
        for item in data_list:
            item['cluster_category'] = f"General {intent_name}"
            item['cluster_reasoning'] = "Not enough data"
        return data_list
        
    texts = [x['text'] for x in data_list]
    
    # 1. Embeddings
    if provider == "OpenAI":
        embeddings, success = get_openai_embeddings(texts, api_key)
    else:
        embeddings, success = get_gemini_embeddings(texts, api_key)
    
    # If embeddings fail, fallback to a "General" group
    if success < 0.5 or len(embeddings) == 0:
        for item in data_list:
            item['cluster_category'] = "Unclustered"
            item['cluster_reasoning'] = "Embedding Generation Failed"
        return data_list

    # 2. KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    
    # 3. Smart Labeling (Sample Representative Tickets)
    samples_map = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        # Get up to 5 examples
        sample_idxs = np.random.choice(indices, min(len(indices), 5), replace=False)
        samples_map[f"{i}"] = [texts[idx][:200] for idx in sample_idxs]
        
    prompt = f"""
    You are analyzing specific themes within '{intent_name}'.
    Task: Create a short 3-5 word Title for each group based on the sample tickets.
    
    Input Groups:
    {json.dumps(samples_map)}

    OUTPUT JSON FORMAT:
    {{ 
      "0": {{ "category": "Login Errors", "reasoning": "Issues with password reset" }},
      "1": {{ "category": "UI Glitches", "reasoning": "Buttons overlapping" }}
    }}
    """
    
    try:
        if provider == "OpenAI":
            response_text = call_openai_json(api_key, prompt)
        else:
            genai.configure(api_key=api_key)
            # 🟢 FIXED: Correct indentation and using stable model
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            response_text = clean_json_response(resp.text)
            
        labels_map = json.loads(response_text)
    except Exception as e:
        print(f"Labeling failed: {e}")
        labels_map = {}

    # 4. Apply Labels
    # Normalize keys
    normalized_labels = {}
    if isinstance(labels_map, dict):
        for k, v in labels_map.items():
            clean_key = re.sub(r"\D", "", str(k))
            normalized_labels[clean_key] = v

    for idx, cluster_id in enumerate(labels):
        c_id_str = str(cluster_id)
        info = normalized_labels.get(c_id_str)
        
        if info:
            cat = info.get('category', f"Group {cluster_id}")
            reason = info.get('reasoning', "AI Labeled")
        else:
            cat = f"Group {cluster_id}"
            reason = "Auto-clustered"
        
        data_list[idx]['cluster_category'] = cat
        data_list[idx]['cluster_reasoning'] = reason
        
    return data_list