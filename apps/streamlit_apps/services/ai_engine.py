import json
import time
import re
import numpy as np
import google.generativeai as genai
from sklearn.cluster import KMeans
import streamlit as st

# --- HELPER: CLEAN AI JSON ---
def clean_json_response(text):
    """
    Removes markdown code blocks (```json ... ```) which cause crashes.
    """
    # Remove code block markers
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# --- ROUTING ---
def classify_batch(requests_batch, model):
    batch_text = ""
    for req in requests_batch:
        # Use first 400 chars of transcript
        clean_text = req['text'][:400].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    prompt = f"""
    You are a Support Ticket Router.
    Classify these transcripts into: 'how_to', 'problem_report', or 'request'.
    
    ### INPUT DATA
    {batch_text}

    ### OUTPUT INSTRUCTIONS
    1. Return strictly valid JSON.
    2. Format: {{ "ID_123": {{ "intent": "...", "explanation": "..." }} }}
    3. Keep explanation short (max 6 words).
    """
    
    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        # CLEAN THE RESPONSE BEFORE PARSING
        clean_text = clean_json_response(response.text)
        return json.loads(clean_text)
    except Exception as e:
        print(f"Routing Batch Failed: {e}")
        return {}

def run_routing(cleaned_data, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    buckets = {"how_to": [], "problem_report": [], "request": []}
    batch_size = 20 # Smaller batch size for better reliability
    progress_bar = st.progress(0)
    total = len(cleaned_data)
    
    for i in range(0, total, batch_size):
        batch = cleaned_data[i : i + batch_size]
        if not batch: continue
        
        results_map = classify_batch(batch, model)
        
        # Handle List vs Dict return structure
        if isinstance(results_map, list):
            temp = {}
            for item in results_map:
                if isinstance(item, dict):
                    # Try to find a key that looks like an ID
                    for k, v in item.items():
                        if k.startswith("ID_"): temp[k] = v
            results_map = temp

        for req in batch:
            key = f"ID_{req['request_id']}"
            res = results_map.get(key, {})
            
            # Default to problem_report if AI failed for this specific row
            intent = res.get("intent", "problem_report")
            explanation = res.get("explanation", "Auto-routed based on keywords")
            
            if intent not in buckets: intent = "problem_report"
            
            req['intent'] = intent
            req['explanation'] = explanation # <--- Ensuring this gets saved
            buckets[intent].append(req)
            
        progress_bar.progress(min((i + batch_size) / total, 1.0))
        time.sleep(1)
        
    return buckets

# --- CLUSTERING ---
def get_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    embeddings = []
    # Batch embeddings to avoid timeouts
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="clustering"
            )
            embeddings.extend(result['embedding'])
            time.sleep(0.5)
        except:
            # Fallback for errors: Zero vectors
            embeddings.extend([[0.0]*768 for _ in batch])
    return np.array(embeddings)

def cluster_and_label_intent(intent_name, data_list, num_clusters, api_key):
    # Safety: If too few items, skip clustering
    if len(data_list) < num_clusters:
        for item in data_list:
            item['cluster_category'] = f"General {intent_name.replace('_', ' ').title()}"
            item['cluster_reasoning'] = "Not enough data to cluster"
        return data_list
        
    texts = [x['text'] for x in data_list]
    embeddings = get_embeddings(texts, api_key)
    
    # K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    
    # Prepare Samples for Labeling AI
    samples_map = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        # Take up to 5 examples to give AI better context
        sample_idxs = np.random.choice(indices, min(len(indices), 5), replace=False)
        samples_map[f"{i}"] = [texts[idx] for idx in sample_idxs]
        
    # --- LABELING CALL ---
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    prompt = f"""
    You are a Senior Support Analyst. 
    I have grouped '{intent_name}' tickets into clusters.
    
    TASK:
    1. Read the sample transcripts for each group.
    2. Create a specific, short Title (3-5 words) that describes the issue. 
       (e.g., "Login SSO Failures", "API Rate Limiting", "Billing Invoice Requests")
    3. Provide a 1-sentence reasoning.

    INPUT GROUPS:
    """
    for k, v in samples_map.items():
        # Clean text slightly to save tokens
        clean_samples = [t[:300].replace("\n", " ") for t in v]
        prompt += f"\nGroup {k}: {json.dumps(clean_samples)}"

    prompt += """
    
    OUTPUT JSON FORMAT:
    {
        "0": { "category": "...", "reasoning": "..." },
        "1": { "category": "...", "reasoning": "..." }
    }
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        # CLEAN RESPONSE
        clean_text = clean_json_response(response.text)
        labels_map = json.loads(clean_text)
        
        # Helper to safely extract dict if wrapped in list
        if isinstance(labels_map, list):
            temp = {}
            for item in labels_map:
                if isinstance(item, dict): temp.update(item)
            labels_map = temp
            
    except Exception as e:
        print(f"Labeling Failed: {e}")
        labels_map = {}
        
    # Apply labels to data
    for idx, cluster_id in enumerate(labels):
        # We look for key "0" or "Group 0" or int(0)
        c_id_str = str(cluster_id)
        
        info = labels_map.get(c_id_str) or labels_map.get(f"Group {c_id_str}")
        
        if info:
            cat = info.get('category', f"Cluster {cluster_id}")
            reason = info.get('reasoning', "AI labeled")
        else:
            cat = f"{intent_name.title()} Group {cluster_id}"
            reason = "AI Labeling step failed to return key"
        
        data_list[idx]['cluster_category'] = cat
        data_list[idx]['cluster_reasoning'] = reason
        
    return data_list