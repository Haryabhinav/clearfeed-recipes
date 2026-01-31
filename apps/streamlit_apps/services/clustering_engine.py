import json
import time
import numpy as np
import logging
import google.generativeai as genai
import openai
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BATCH_SIZE = 100  # Max texts per API call

def clean_json_response(text):
    """Clean JSON response from LLM output."""
    text = str(text)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

def call_openai(api_key, prompt):
    """Call OpenAI API with the given prompt."""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# ======================================================
# PHASE 1: MATH & EMBEDDINGS (Deterministic)
# ======================================================

def get_embeddings_batched_safe(texts, api_key):
    """
    Generates embeddings in serial batches to respect rate limits.
    Returns L2-normalized numpy array.
    """
    genai.configure(api_key=api_key)
    
    # Pre-cleaning: ensure no empty strings
    clean_texts = [str(t)[:1000].strip() or " " for t in texts]
    total_texts = len(clean_texts)
    all_embeddings = []
    
    logger.info(f"Initiating Gemini Embeddings for {total_texts} items in batches of {BATCH_SIZE}...")

    # Process in chunks
    for i in range(0, total_texts, BATCH_SIZE):
        batch = clean_texts[i : i + BATCH_SIZE]
        batch_embeddings = None
        
        # Retry Logic (Max 1 retry per batch)
        for attempt in range(2):
            try:
                # API Call
                result = genai.embed_content(
                    model="models/text-embedding-004", 
                    content=batch,
                    task_type="clustering"
                )
                
                # Validation
                if 'embedding' in result:
                    vectors = result['embedding']
                    if len(vectors) == len(batch):
                        batch_embeddings = vectors
                        break
                    else:
                        logger.warning(f"Batch {i}: Shape mismatch (Sent {len(batch)}, Got {len(vectors)})")
            except Exception as e:
                logger.warning(f"Batch {i} failed (Attempt {attempt+1}): {e}")
                time.sleep(1)

        if batch_embeddings is None:
            # CRITICAL FAILURE: Stop pipeline. Do not return zeros.
            raise RuntimeError(f"Embedding failed completely for batch index {i}. Pipeline halted.")
        
        all_embeddings.extend(batch_embeddings)

    # Convert to Numpy
    emb_matrix = np.array(all_embeddings)

    # Health Checks
    norms = np.linalg.norm(emb_matrix, axis=1)
    near_zero = norms < 1e-9
    if np.any(near_zero):
        pct = np.mean(near_zero) * 100
        logger.warning(f"⚠️ {pct:.2f}% of vectors are near-zero. (Data loss)")
        # Fix zeros to avoid NaN in normalization
        emb_matrix[near_zero] = 1e-10

    # L2 Normalization (Crucial for Euclidean distance used in Ward linkage)
    normalized_matrix = normalize(emb_matrix, norm='l2')
    
    return normalized_matrix

def cluster_texts_agglomerative(texts, embeddings, distance_threshold=1.2):
    """
    Performs Agglomerative Hierarchical Clustering.
    Automatically determines number of clusters based on distance_threshold.
    
    Args:
        texts: List of strings.
        embeddings: Numpy array of embeddings.
        distance_threshold: The linkage distance threshold above which, 
                            clusters will not be merged.
    """
    # 1. Define and Fit the Model
    # Note: n_clusters=None forces the model to use distance_threshold
    model = AgglomerativeClustering(
        n_clusters=None,           
        distance_threshold=distance_threshold,
        linkage='ward',            
        metric='euclidean',        
        compute_full_tree=True     
    )
    
    labels = model.fit_predict(embeddings)
    n_clusters_found = model.n_clusters_
    logger.info(f"Agglomerative Clustering found {n_clusters_found} clusters (Threshold: {distance_threshold})")

    # 2. Find representative samples
    # Agglomerative clustering does not calculate centroids during fit.
    # We calculate the arithmetic mean of the cluster points to act as a pseudo-centroid.
    
    cluster_info = {}
    
    for cluster_id in range(n_clusters_found):
        mask = (labels == cluster_id)
        indices = np.where(mask)[0]
        cluster_points = embeddings[mask]
        
        if len(cluster_points) > 0:
            # Calculate manual centroid for this cluster
            centroid = np.mean(cluster_points, axis=0).reshape(1, -1)
            
            # Find point closest to this manual centroid
            closest_idx_in_cluster, _ = pairwise_distances_argmin_min(centroid, cluster_points)
            # Map back to original index
            centroid_original_idx = indices[closest_idx_in_cluster[0]]
            
            # Get random other indices for variety (excluding the centroid)
            other_indices = indices[indices != centroid_original_idx]
            
            random_picks = []
            if len(other_indices) > 0:
                count = min(4, len(other_indices))
                random_picks = np.random.choice(other_indices, count, replace=False)
                
            selected_idx = np.concatenate(([centroid_original_idx], random_picks)).astype(int)
            
            # Format samples
            samples = [texts[i][:1000].replace("\n", " ") for i in selected_idx]
            cluster_info[str(cluster_id)] = samples
        else:
            cluster_info[str(cluster_id)] = ["No samples found"]
        
    return labels, cluster_info

# ======================================================
# PHASE 2: ORCHESTRATION & LABELING
# ======================================================

def generate_cluster_labels(intent_name, cluster_samples, api_key, provider, call_ai_func):
    """
    Single LLM call to label all clusters based on their samples.
    """
    prompt = f"""
    You are analyzing {len(cluster_samples)} DISTINCT groups of '{intent_name}' tickets.
    
    DATA SAMPLES:
    {json.dumps(cluster_samples)}
    
    CRITICAL RULES:
    1. Give each group a UNIQUE title (3-5 words).
    2. Focus on the *intent* shown in the samples.
    3. ABSOLUTELY NO DUPLICATE TITLES.
    
    OUTPUT JSON: {{ "0": "Title A", "1": "Title B" }}
    """

    try:
        if provider == "OpenAI":
            response_text = call_ai_func(api_key, prompt)
        else:
            # Gemini Generation Fallback
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash") 
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            response_text = resp.text
            
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        logger.error(f"Labeling failed: {e}")
        # Fallback labels if LLM fails
        return {k: f"{intent_name} Group {k}" for k in cluster_samples.keys()}

# ======================================================
# MAIN FUNCTION (The Drop-in Replacement)
# ======================================================
# ======================================================
# UPDATED MAIN FUNCTION (Threshold Hardcoded to 1.0)
# ======================================================
def cluster_and_label_intent(intent_name, data_list, distance_threshold, api_key, provider="Gemini", call_ai_func=None, clean_json_func=None):
    """
    Main function to cluster and label intents.
    """
    
    # Defaults
    if call_ai_func is None: call_ai_func = call_openai
    if clean_json_func is None: clean_json_func = clean_json_response

    # 1. Validation
    if len(data_list) < 3:
        for item in data_list: item['cluster_category'] = f"General {intent_name}"
        return data_list
        
    texts = [x['text'] for x in data_list]
    
    # --- FORCE THRESHOLD TO 1.0 ---
    # Ignoring the input argument 'distance_threshold' completely.
    distance_threshold = 1.2
    # ------------------------------

    try:
        # 2. Math Phase: Embed & Cluster (Agglomerative)
        embeddings = get_embeddings_batched_safe(texts, api_key)
        
        labels_idx, cluster_samples = cluster_texts_agglomerative(
            texts, 
            embeddings, 
            distance_threshold=distance_threshold
        )
        
        # 3. Labeling Phase: LLM
        labels_map = generate_cluster_labels(intent_name, cluster_samples, api_key, provider, call_ai_func)
        
        # 4. Merge Results
        for idx, cluster_id in enumerate(labels_idx):
            cid_str = str(cluster_id)
            name = labels_map.get(cid_str, f"Group {cid_str}")
            data_list[idx]['cluster_category'] = name
            
        return data_list

    except RuntimeError as e:
        logger.error(f"Pipeline Error: {e}")
        for item in data_list: item['cluster_category'] = "Uncategorized (Error)"
        return data_list