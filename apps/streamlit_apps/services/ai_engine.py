# File: services/ai_engine.py

import json
import time
import re
import numpy as np
from sklearn.cluster import KMeans
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import openai
import streamlit as st

# --- CONFIG ---
# Gemini Free Tier Limit: 15 RPM = 1 request every 4 seconds.
# We set a safe delay to avoid "429 Resource Exhausted" errors.
GEMINI_SAFE_DELAY = 4.0 

# --- HELPER: CLEAN AI JSON ---
def clean_json_response(text):
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# --- OPENAI SPECIFIC HELPERS ---
def call_openai_json(api_key, prompt, model="gpt-4o"):
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
        st.warning("⚠️ OpenAI Rate Limit Hit. Sleeping for 20s...")
        time.sleep(20)
        return "{}"
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return "{}"

# --- GEMINI EMBEDDINGS (With Strict Rate Limiting) ---
# --- GEMINI EMBEDDINGS (With Empty Content Fallback) ---
def get_gemini_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    embeddings = []
    
    # Process in small batches
    for i in range(0, len(texts), 20):
        batch = texts[i:i+20]
        
        # --- FALLBACK PROTECTION ---
        # 1. Sanitize the batch: The API crashes if any single item is empty.
        # 2. We preserve order by replacing empty text with a placeholder.
        safe_batch = []
        for t in batch:
            # Truncate and strip whitespace
            cleaned = t[:500].strip() 
            if not cleaned:
                safe_batch.append("empty_content") # Placeholder text
            else:
                safe_batch.append(cleaned)

        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=safe_batch, 
                task_type="clustering"
            )
            embeddings.extend(result['embedding'])
            
            # Rate limit protection
            time.sleep(2.0) 

        except Exception as e:
            print(f"Embedding batch failed: {e}")
            # Fallback: If the API call still fails, insert Zero Vectors 
            # so the list length matches the data length (Critical for K-Means)
            # Gemini embedding-004 has 768 dimensions.
            zero_vectors = [[0.0] * 768 for _ in range(len(batch))]
            embeddings.extend(zero_vectors)
            
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
            return [], 0.0
    return np.array(embeddings), 1.0

# --- 1. CLASSIFICATION (ROUTING) ---
def classify_batch(requests_batch, api_key, provider):
    batch_text = ""
    for req in requests_batch:
        clean_text = req['text'][:200].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    prompt = f"""
    You are a Support Ticket Classifier.
    Task: Label each ticket with exactly ONE category.
    
    ### DEFINITIONS (STRICTLY FOLLOW THESE):
    
    1. **feature_request**: 
       - The customer requests a new feature/enhancement AND the agent explicitly confirms it as a potential improvement.
       - If Agent says it already exists, it is NOT a feature_request.
       
    3. **how_to_question**: 
       - Customer asks for clarification or instruction on how to use a feature.
       
    4. **problem_report**: 
        - Customer reports problems with features/services (broader than bug).
        - If Agent confirms it's a bug, it is a problem_report.
    Input Data:
    {batch_text}
    
    OUTPUT JSON FORMAT (Strictly Key-Value):
    {{
      "ID_123": "feature_request",
      "ID_456": "problem_report"
    }}
    """
    
    if provider == "OpenAI":
        return json.loads(call_openai_json(api_key, prompt))
    else:
        genai.configure(api_key=api_key)
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
        "unclassified": [] 
    }
    
    # Use slightly larger batch size to reduce number of API calls
    batch_size = 30 
    progress_bar = st.progress(0)
    total = len(cleaned_data)
    
    for i in range(0, total, batch_size):
        batch = cleaned_data[i : i + batch_size]
        if not batch: continue
        
        results_map = classify_batch(batch, api_key, provider)
        
        # --- RATE LIMIT PROTECTION ---
        if provider == "Gemini":
            time.sleep(GEMINI_SAFE_DELAY) # Sleep 4s to stay under 15 RPM
        
        # Normalize Keys (Handle ID_123 vs 123)
        normalized_results = {}
        if results_map:
            for k, v in results_map.items():
                clean_key = re.sub(r"\D", "", str(k))
                normalized_results[clean_key] = v

        for req in batch:
            req_id_str = str(req['request_id'])
            res = normalized_results.get(req_id_str)
            
            raw_label = "unclassified"
            explanation = "AI could not classify"
            
            if isinstance(res, str):
                raw_label = res
                explanation = "AI Classified"
            elif isinstance(res, dict):
                raw_label = res.get("label", "unclassified")
                explanation = res.get("explanation", "AI Classified")
            
            raw_label = raw_label.lower().strip()
            if "feature" in raw_label: raw_label = "feature_request"
            elif "problem" in raw_label or "bug" in raw_label: raw_label = "problem_report"
            elif "how" in raw_label or "question" in raw_label: raw_label = "how_to_question"
            
            if raw_label in buckets:
                req['intent'] = raw_label
                req['explanation'] = explanation
                buckets[raw_label].append(req)
            else:
                req['intent'] = "unclassified"
                req['explanation'] = f"Raw AI output: {raw_label}"
                buckets["unclassified"].append(req)
            
        progress_bar.progress(min((i + batch_size) / total, 1.0))
        
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
    
    if success < 0.5 or len(embeddings) == 0:
        for item in data_list:
            item['cluster_category'] = "Unclustered"
            item['cluster_reasoning'] = "Embedding Generation Failed"
        return data_list

    # 2. KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    
    # 3. Smart Labeling
    samples_map = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
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
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            
            # RATE LIMIT PROTECTION
            time.sleep(GEMINI_SAFE_DELAY) 
            
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            response_text = clean_json_response(resp.text)
            
        labels_map = json.loads(response_text)
    except Exception as e:
        print(f"Labeling failed: {e}")
        labels_map = {}

    # 4. Apply Labels
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