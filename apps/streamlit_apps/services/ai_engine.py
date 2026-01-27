import json
import time
import re
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import google.generativeai as genai
import openai
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- STRICT QUOTA CONFIG ---
ROUTING_BATCH_SIZE = 200 
EMBEDDING_BATCH_SIZE = 100
MAX_WORKERS = 10 

# --- HELPER: CLEAN AI JSON ---
def clean_json_response(text):
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# --- NEW: HEURISTIC CLUSTERING FUNCTION ---
def auto_discover_clusters_heuristic(embeddings, max_k=8):
    """
    Calculates K using the formula: k ≈ sqrt(N/2).
    Uses MiniBatchKMeans for maximum speed.
    """
    n_samples = len(embeddings)
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), 1
    
    raw_k = int(np.sqrt(n_samples / 2))
    final_k = max(2, min(raw_k, max_k, n_samples))
    
    kmeans = MiniBatchKMeans(
        n_clusters=final_k, 
        random_state=42, 
        n_init=3
    ).fit(embeddings)
    
    return kmeans.labels_, final_k

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
    except Exception:
        return "{}"

# --- GEMINI EMBEDDINGS (MAX PARALLELISM) ---
def get_gemini_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    batches = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batches.append(texts[i:i+EMBEDDING_BATCH_SIZE])
    
    embeddings_map = {} 
    
    def process_embedding_batch(batch_idx, batch_data):
        safe_batch = []
        for t in batch_data:
            cleaned = t[:200].strip()
            safe_batch.append(cleaned if cleaned else "empty_content")
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=safe_batch, 
                task_type="clustering"
            )
            return batch_idx, result['embedding']
        except Exception as e:
            print(f"Embedding batch {batch_idx} failed: {e}")
            zeros = [[0.0] * 768 for _ in range(len(batch_data))]
            return batch_idx, zeros

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_embedding_batch, i, b) for i, b in enumerate(batches)]
        for future in as_completed(futures):
            idx, result = future.result()
            embeddings_map[idx] = result
            
    final_embeddings = []
    for i in range(len(batches)):
        final_embeddings.extend(embeddings_map.get(i, []))
    return np.array(final_embeddings), 1.0

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
        clean_text = req['text'][:120].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    prompt = f"""
    Task: Label each ticket ID.
    DEFINITIONS:
    1. feature_request: Request for new feature.
    2. bug: Broken functionality.
    3. how_to_question: Asking for instructions.
    4. problem_report: General issues.
    5. request: General help.
    INPUT:
    {batch_text}
    OUTPUT JSON:
    {{ "ID_123": {{ "label": "feature_request", "explanation": "..." }} }}
    """
    try:
        if provider == "OpenAI":
            return json.loads(call_openai_json(api_key, prompt))
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash") 
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(clean_json_response(response.text))
    except Exception as e:
        print(f"Batch failed: {e}")
        return {}

def run_routing(cleaned_data, api_key, provider="Gemini"):
    buckets = {"feature_request": [], "problem_report": [], "how_to_question": [], "unclassified": []}
    batches = [cleaned_data[i : i + ROUTING_BATCH_SIZE] for i in range(0, len(cleaned_data), ROUTING_BATCH_SIZE)]
    normalized_results = {}
    progress_bar = st.progress(0)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_batch, batch, api_key, provider): batch for batch in batches}
        for future in as_completed(futures):
            results_map = future.result()
            if results_map:
                for k, v in results_map.items():
                    clean_key = re.sub(r"\D", "", str(k))
                    normalized_results[clean_key] = v
            completed += 1
            progress_bar.progress(completed / len(batches))

    for req in cleaned_data:
        res = normalized_results.get(str(req['request_id']))
        raw_label = res.get("label", "unclassified").lower().strip() if isinstance(res, dict) else "unclassified"
        
        if "feature" in raw_label or raw_label == "request": bucket = "feature_request"
        elif "bug" in raw_label or "problem" in raw_label: bucket = "problem_report"
        elif "how" in raw_label or "question" in raw_label: bucket = "how_to_question"
        else: bucket = "unclassified"
        
        req['intent'] = bucket
        req['explanation'] = res.get("explanation", "") if isinstance(res, dict) else ""
        buckets[bucket].append(req)
    return buckets

# --- 2. CLUSTERING FUNCTION (UPDATED WITH HEURISTIC) ---
def cluster_and_label_intent(intent_name, data_list, num_clusters, api_key, provider="Gemini"):
    min_data_for_clustering = 5
    if len(data_list) < min_data_for_clustering:
        for item in data_list:
            item['cluster_category'] = f"General {intent_name}"
            item['cluster_reasoning'] = "Not enough data for clustering"
        return data_list
        
    texts = [x['text'] for x in data_list]
    if provider == "OpenAI":
        embeddings, success = get_openai_embeddings(texts, api_key)
    else:
        embeddings, success = get_gemini_embeddings(texts, api_key)
    
    if success < 0.5 or len(embeddings) == 0:
        for item in data_list:
            item['cluster_category'] = "Unclustered"
            item['cluster_reasoning'] = "Embedding Generation Failed"
        return data_list

    # --- UPDATED: Heuristic K-Selection Logic ---
    if not num_clusters or num_clusters == 0:
        labels, final_k = auto_discover_clusters_heuristic(embeddings, max_k=8)
    else:
        final_k = num_clusters
        kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10).fit(embeddings)
        labels = kmeans.labels_
    
    # 4. Smart Labeling
    samples_map = {}
    for i in range(final_k):
        indices = np.where(labels == i)[0]
        sample_idxs = np.random.choice(indices, min(len(indices), 5), replace=False)
        samples_map[f"{i}"] = [texts[idx][:150] for idx in sample_idxs]
        
    prompt = f"Analyze '{intent_name}'. Task: Create 3-5 word Title for each group. Input: {json.dumps(samples_map)} Output JSON: {{ '0': {{ 'category': '...', 'reasoning': '...' }} }}"
    
    try:
        if provider == "OpenAI":
            response_text = call_openai_json(api_key, prompt)
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            response_text = clean_json_response(resp.text)
        labels_map = json.loads(response_text)
    except Exception:
        labels_map = {}

    normalized_labels = {re.sub(r"\D", "", str(k)): v for k, v in labels_map.items()} if isinstance(labels_map, dict) else {}

    for idx, cluster_id in enumerate(labels):
        info = normalized_labels.get(str(cluster_id))
        data_list[idx]['cluster_category'] = info.get('category', f"Group {cluster_id}") if info else f"Group {cluster_id}"
        data_list[idx]['cluster_reasoning'] = info.get('reasoning', "Auto-clustered") if info else "Auto-clustered"
        
    return data_list