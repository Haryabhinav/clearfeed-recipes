import json
import re
import logging
import numpy as np
import google.generativeai as genai
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Added for progress tracking

logger = logging.getLogger(__name__)

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
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

ROUTING_BATCH_SIZE = 50 
MAX_WORKERS = 10

def fast_route(text):
    t = text.lower()
    return None

def classify_batch(requests_batch, api_key, provider, call_openai_func, clean_json_func):
    batch_text = ""
    for req in requests_batch:
        clean_text = req['text'][:500].replace("\n", " ").replace('"', "'")
        batch_text += f"ID_{req['request_id']}: {clean_text}\n"

    prompt = f"""
    You are an Expert Support Triage Agent.
    Task: Classify the following {len(requests_batch)} conversation snippets into exactly ONE of these labels:
    [feature_request,how_to_question, problem_report]

    ### DEFINITIONS & RULES (Strict Adherence):
    
    1. **feature_request**: 
       - User asks for a new feature/enhancement.
       - *Condition:* Ideally confirmed by Agent ("We will look into making this enhancement").
       - If Agent says "It already exists", label as 'how_to_question'.
       - General request for assistance (e.g., "Need a separate account", "Update my billing").

    2. **how_to_question**: 
       - User asks for clarification or instructions (e.g., "How do I...", "Where is...").
       - Includes cases where User asks for a feature but Agent explains it already exists.

    3. **problem_report**: 
       - General report of issues/service problems.
       - Use this if it looks like a bug but the Agent hasn't explicitly confirmed it yet, or if it's a configuration error.
       - User reports a broken logic or error.
       - *Condition:* Agent admits the issue ("Something def. wrong, we are looking into it").
       - Must be something the user cannot fix themselves.


    ### OUTPUT FORMAT:
    Return strictly a JSON object mapping ID to Label. No explanations.
    {{
        "ID_123": "problem_report",
        "ID_456": "feature_request"
    }}

    ### INPUT CONVERSATIONS:
    {batch_text}
    """
    
    try:
        if provider == "OpenAI":
            return json.loads(call_openai_func(api_key, prompt))
        else:
            genai.configure(api_key=api_key)
            # Using 1.5-flash as the stable current version
            model = genai.GenerativeModel("gemini-2.5-flash") 
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(clean_json_func(response.text))
    except Exception as e:
        logger.error(f"Classification Batch Failed: {e}")
        return {}

def run_routing(cleaned_data, api_key, provider, call_openai_func=None, clean_json_func=None):
    # Use default functions if not provided
    if call_openai_func is None:
        call_openai_func = call_openai
    if clean_json_func is None:
        clean_json_func = clean_json_response

    buckets = {"feature_request": [], "problem_report": [], "how_to_question": [], "unclassified": []}
    needs_llm = []

    # --- Phase 1: Regex Fast Routing ---
    logger.info("Starting Regex fast-routing...")
    for req in tqdm(cleaned_data, desc="🔍 Fast Routing", unit="ticket"):
        label = fast_route(req["text"])
        if label:
            req['intent'] = label
            buckets[label].append(req)
        else:
            needs_llm.append(req)

    if not needs_llm:
        return buckets

    # --- Phase 2: LLM Batch Routing ---
    batches = [needs_llm[i : i + ROUTING_BATCH_SIZE] for i in range(0, len(needs_llm), ROUTING_BATCH_SIZE)]
    normalized_results = {}

    logger.info(f"Sending {len(needs_llm)} tickets to {provider} for deep classification...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_batch, batch, api_key, provider, call_openai_func, clean_json_func): batch for batch in batches}

        # Progress bar for LLM batch completion
        for future in tqdm(as_completed(futures), total=len(batches), desc=f"🤖 {provider} Routing", unit="batch"):
            results_map = future.result()
            if results_map:
                for k, v in results_map.items():
                    normalized_results[re.sub(r"\D", "", str(k))] = v

    # Final Assignment
    for req in needs_llm:
        res = normalized_results.get(str(req['request_id']), "unclassified")
        raw_label = res.lower() if isinstance(res, str) else "unclassified"

        if "feature" in raw_label or raw_label == "request": bucket = "feature_request"
        elif "bug" in raw_label or "problem" in raw_label: bucket = "problem_report"
        elif "how" in raw_label or "question" in raw_label: bucket = "how_to_question"
        else: bucket = "unclassified"

        req['intent'] = bucket
        buckets[bucket].append(req)

    logger.info("Routing complete.")
    return buckets