import json
import time
import numpy as np
import google.generativeai as genai
from sklearn.cluster import KMeans
import streamlit as st

# --- ROUTING ---
def classify_batch(requests_batch, model):
    batch_text = ""
    for req in requests_batch:
        # 400 chars context
        clean_text = req['text'][:400].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    prompt = f"""
    You are a Support Ticket Router. Classify these transcripts.
    Input format: "User: ... Agent: ...".
    Ignore greetings. Focus on the core request.

    ### DEFINITIONS
    1. **how_to**
       * Definition: Asking for docs, instructions, or "how do I".
    2. **problem_report**
       * Definition: Errors, bugs, "not working", failures.
    3. **request**
       * Definition: Feature requests, upgrades, admin tasks.

    ### INPUT DATA
    {batch_text}

    ### OUTPUT JSON
    {{ "ID_123": {{ "intent": "how_to", "explanation": "Asking for docs" }} }}
    """
    
    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Routing failed: {e}")
        return {}

def run_routing(cleaned_data, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    buckets = {"how_to": [], "problem_report": [], "request": []}
    batch_size = 30 
    progress_bar = st.progress(0)
    total = len(cleaned_data)
    
    for i in range(0, total, batch_size):
        batch = cleaned_data[i : i + batch_size]
        if not batch: continue
        
        results_map = classify_batch(batch, model)
        
        # Safety: List to Dict conversion
        if isinstance(results_map, list):
            temp = {}
            for x in results_map:
                if isinstance(x, dict): temp.update(x)
            results_map = temp

        for req in batch:
            key = f"ID_{req['request_id']}"
            res = results_map.get(key, {})
            intent = res.get("intent", "problem_report")
            if intent not in buckets: intent = "problem_report"
            
            req['intent'] = intent
            req['explanation'] = res.get("explanation", "")
            buckets[intent].append(req)
            
        progress_bar.progress(min((i + batch_size) / total, 1.0))
        time.sleep(1)
        
    return buckets

# --- CLUSTERING ---
def get_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="clustering"
            )
            embeddings.extend(result['embedding'])
            time.sleep(1)
        except:
            embeddings.extend([[0.0]*768 for _ in batch])
    return np.array(embeddings)

def cluster_and_label_intent(intent_name, data_list, num_clusters, api_key):
    # Check if we have enough data to cluster
    if len(data_list) < num_clusters:
        # If not enough data, just label them all "General"
        for item in data_list:
            item['cluster_category'] = f"General {intent_name.replace('_', ' ').title()}"
            item['cluster_reasoning'] = "Low volume"
        return data_list
        
    texts = [x['text'] for x in data_list]
    embeddings = get_embeddings(texts, api_key)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    
    # Pick Samples
    samples_map = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        sample_idxs = np.random.choice(indices, min(len(indices), 3), replace=False)
        samples_map[f"{intent_name}::{i}"] = [texts[idx] for idx in sample_idxs]
        
    # --- LABELING (Fixed for ALL clusters) ---
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    prompt = f"""
    You are an expert Analyst. 
    I will provide groups of transcripts from the intent category: '{intent_name}'.
    
    TASK:
    1. Read the samples.
    2. Give a short, specific Label (3-5 words) for the cluster.
    3. Do NOT use generic names like "Cluster 0". Be descriptive.

    OUTPUT JSON:
    {{
        "{intent_name}::0": {{ "category": "Login Issues", "reasoning": "..." }},
        "{intent_name}::1": {{ "category": "API Docs", "reasoning": "..." }}
    }}
    """
    
    for k, v in samples_map.items():
        prompt += f"\nGroup {k}:\n" + "\n".join([f"- {t[:400]}" for t in v])
        
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        labels_map = json.loads(response.text)
        
        if isinstance(labels_map, list):
            temp = {}
            for x in labels_map:
                if isinstance(x, dict): temp.update(x)
            labels_map = temp
            
    except Exception as e:
        print(f"Labeling failed: {e}")
        labels_map = {}
        
    # Apply labels
    for idx, cluster_id in enumerate(labels):
        key = f"{intent_name}::{cluster_id}"
        # Fallback if AI missed a key
        default_cat = f"{intent_name.replace('_',' ').title()} Group {cluster_id}"
        
        info = labels_map.get(key, {})
        data_list[idx]['cluster_category'] = info.get('category', default_cat)
        data_list[idx]['cluster_reasoning'] = info.get('reasoning', "Auto-generated")
        
    return data_list