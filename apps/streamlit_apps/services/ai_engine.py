import json
import time
import re
import numpy as np
from sklearn.cluster import KMeans
import google.generativeai as genai
import openai
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- STRICT QUOTA CONFIG ---
# RPD Limit is 20. We utilize max batch sizes to keep total calls low.
# Routing: 200 tickets/call -> ~4 calls total (Safe for 5 RPM limit)
# Embeddings: 100 tickets/call -> ~8 calls total (Embedding API usually has higher limits)
ROUTING_BATCH_SIZE = 200 
EMBEDDING_BATCH_SIZE = 100
MAX_WORKERS = 10  # Enough threads to fire ALL requests simultaneously

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
    except Exception:
        return "{}"

# --- GEMINI EMBEDDINGS (MAX PARALLELISM) ---
def get_gemini_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    
    # 1. Prepare all batches upfront
    batches = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batches.append(texts[i:i+EMBEDDING_BATCH_SIZE])
    
    embeddings_map = {} 
    
    def process_embedding_batch(batch_idx, batch_data):
        safe_batch = []
        for t in batch_data:
            # Truncate to 200 chars to ensure speed
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

    # 2. Fire ALL batches at the exact same moment
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_embedding_batch, i, b) for i, b in enumerate(batches)]
        
        for future in as_completed(futures):
            idx, result = future.result()
            embeddings_map[idx] = result
            
    # 3. Reassemble in order
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
        # OPTIMIZATION: Truncate to 200 chars. 
        # Reduces token count massively, allowing the AI to process 200 items quickly.
        clean_text = req['text'][:120].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    prompt = f"""
    Task: Label each ticket ID.
    
    DEFINITIONS:
    1. feature_request: Request for new feature (Confirmed by Agent).
    2. bug: Broken functionality (Confirmed by Agent).
    3. how_to_question: Asking for instructions.
    4. problem_report: General issues.
    5. request: General help.

    INPUT:
    {batch_text}
    
    OUTPUT JSON (Strictly key-value, explanation < 5 words):
    {{
      "ID_123": {{ "label": "feature_request", "explanation": "Agent confirmed request" }}
    }}
    """
    
    try:
        if provider == "OpenAI":
            return json.loads(call_openai_json(api_key, prompt))
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash") 
            # No Stream=True needed for batch, just standard generation
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(clean_json_response(response.text))
    except Exception as e:
        print(f"Batch failed: {e}")
        return {}

def run_routing(cleaned_data, api_key, provider="Gemini"):
    buckets = {
        "feature_request": [],
        "problem_report": [],
        "how_to_question": [],
        "unclassified": [] 
    }
    
    # 1. Prepare Batches
    batches = []
    for i in range(0, len(cleaned_data), ROUTING_BATCH_SIZE):
        batches.append(cleaned_data[i : i + ROUTING_BATCH_SIZE])

    total_batches = len(batches)
    normalized_results = {}
    
    progress_bar = st.progress(0)
    completed = 0
    
    # 2. MAXIMIZE PARALLELISM: Fire ALL routing requests instantly
    # With batch size 200, 800 tickets = 4 requests.
    # 4 requests < 5 RPM limit. This is safe to run in parallel.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_batch, batch, api_key, provider): batch for batch in batches}
        
        for future in as_completed(futures):
            results_map = future.result()
            
            if results_map:
                for k, v in results_map.items():
                    clean_key = re.sub(r"\D", "", str(k))
                    normalized_results[clean_key] = v
            
            completed += 1
            progress_bar.progress(completed / total_batches)

    # 3. Map Results
    for req in cleaned_data:
        req_id_str = str(req['request_id'])
        res = normalized_results.get(req_id_str)
        
        raw_label = "unclassified"
        explanation = "AI could not classify"
        
        if isinstance(res, dict):
            raw_label = res.get("label", "unclassified")
            explanation = res.get("explanation", "")
        elif isinstance(res, str):
            raw_label = res
        
        raw_label = raw_label.lower().strip()
        
        # MAPPING LOGIC
        if "feature" in raw_label or raw_label == "request": 
            bucket = "feature_request"
        elif "bug" in raw_label or "problem" in raw_label: 
            bucket = "problem_report"
        elif "how" in raw_label or "question" in raw_label: 
            bucket = "how_to_question"
        else: 
            bucket = "unclassified"
        
        req['intent'] = bucket
        req['explanation'] = explanation
        buckets[bucket].append(req)
            
    return buckets

# --- 2. CLUSTERING FUNCTION ---
def cluster_and_label_intent(intent_name, data_list, num_clusters, api_key, provider="Gemini"):
    if len(data_list) < num_clusters:
        for item in data_list:
            item['cluster_category'] = f"General {intent_name}"
            item['cluster_reasoning'] = "Not enough data"
        return data_list
        
    texts = [x['text'] for x in data_list]
    
    # 1. Embeddings (Now Parallelized)
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
    # This only runs 3 times total (once per bucket), so parallelizing this isn't strictly necessary for speed
    # but we keep context minimal.
    samples_map = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        sample_idxs = np.random.choice(indices, min(len(indices), 5), replace=False)
        samples_map[f"{i}"] = [texts[idx][:150] for idx in sample_idxs]
        
    prompt = f"""
    Analyze '{intent_name}'.
    Task: Create 3-5 word Title for each group.
    
    Input:
    {json.dumps(samples_map)}

    OUTPUT JSON:
    {{ "0": {{ "category": "Login Errors", "reasoning": "..." }} }}
    """
    
    try:
        if provider == "OpenAI":
            response_text = call_openai_json(api_key, prompt)
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
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