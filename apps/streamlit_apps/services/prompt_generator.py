import json
import random
import pandas as pd
import google.generativeai as genai
import openai
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- HELPER: CLEAN AI JSON ---
def clean_json_response(text):
    text = str(text)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# --- LLM HELPERS ---
def generate_descriptions(api_key, provider, cluster_data):
    """
    Generates clear descriptions for each cluster to use in the system prompt.
    """
    prompt_input = {}
    for intent, clusters in cluster_data.items():
        prompt_input[intent] = {}
        for cluster_name, examples in clusters.items():
            prompt_input[intent][cluster_name] = examples[:3]

    prompt = f"""
    Task: Write a 1-sentence clear definition for each ticket category based on the examples.
    
    Input:
    {json.dumps(prompt_input)}
    
    OUTPUT JSON:
    {{
      "Feature Requests": {{
         "Integrations": "Requests for connecting with third-party tools like Jira or Slack."
      }}
    }}
    """

    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(clean_json_response(response.text))
    except Exception as e:
        print(f"Description generation failed: {e}")
        return {}

def generate_classification_prompt(df, api_key, provider="Gemini"):
    """
    Generates the definitions markdown (Phase B).
    """
    if df is None or df.empty:
        return "No data available to generate prompts."

    structured_data = {}
    valid_df = df[
        (df['cluster_category'] != "Unclustered") & 
        (~df['cluster_category'].str.startswith("General "))
    ]

    for intent in valid_df['intent'].unique():
        structured_data[intent] = {}
        intent_df = valid_df[valid_df['intent'] == intent]
        for cluster in intent_df['cluster_category'].unique():
            cluster_df = intent_df[intent_df['cluster_category'] == cluster]
            texts = cluster_df['text'].tolist()
            clean_texts = [t.replace("\n", " ")[:200] for t in texts]
            structured_data[intent][cluster] = clean_texts

    descriptions_map = generate_descriptions(api_key, provider, structured_data)

    md_output = f"# Auto-Generated Classification Definitions\n"
    md_output += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    for intent, clusters in structured_data.items():
        if not clusters: continue
        display_intent = intent.replace("_", " ").title()
        md_output += f"## Category: {display_intent}\n\n"
        
        for cluster_name, all_examples in clusters.items():
            desc = descriptions_map.get(intent, {}).get(cluster_name, "Issues related to this specific topic.")
            pos_examples = random.sample(all_examples, min(len(all_examples), 5))
            
            # Sibling negative sampling
            sibling_examples = []
            for sibling_name, sibling_data in clusters.items():
                if sibling_name != cluster_name:
                    sibling_examples.extend(sibling_data)
            neg_examples = []
            if sibling_examples:
                neg_examples = random.sample(sibling_examples, min(len(sibling_examples), 3))
            
            md_output += f"### Sub-category: {cluster_name}\n"
            md_output += f"**Description:** {desc}\n"
            md_output += "**Positive Examples:** " + "; ".join([f'"{p[:100]}..."' for p in pos_examples]) + "\n"
            if neg_examples:
                md_output += "**Negative Examples:** " + "; ".join([f'"{n[:100]}..."' for n in neg_examples]) + "\n"
            md_output += "\n"

    return md_output

# --- PHASE C: INFERENCE & VALIDATION ---

def build_inference_system_prompt(definitions_md):
    """
    Wraps the definitions into the final Production Prompt.
    """
    return f"""
You are an expert support ticket classifier. 
Your goal is to categorize user requests into the correct Category and Sub-category based strictly on the definitions below.

---
DEFINITIONS:
{definitions_md}
---

INSTRUCTIONS:
1. Analyze the input text.
2. Match it to the most similar 'Positive Example' or Description.
3. If it matches a 'Negative Example', do NOT select that sub-category.

OUTPUT FORMAT:
You must output a JSON object with exactly these keys:
{{
  "category": "Exact Category Name",
  "sub_category": "Exact Sub-category Name"
}}
"""

def validate_prompt(df, definitions_md, api_key, provider="Gemini"):
    """
    Runs the generated prompt on 20 random tickets to test accuracy.
    """
    system_prompt = build_inference_system_prompt(definitions_md)
    
    # 1. Select 20 random samples
    if len(df) > 20:
        sample_df = df.sample(20).copy()
    else:
        sample_df = df.copy()
        
    results = []

    def classify_single(text, idx):
        try:
            user_msg = f"Input Text: {text[:300]}"
            
            if provider == "OpenAI":
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={"type": "json_object"}
                )
                raw = resp.choices[0].message.content
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                # Gemini doesn't always support system instructions in standard generate_content, 
                # but we can prepend it to the prompt for robustness.
                full_prompt = f"{system_prompt}\n\n{user_msg}"
                resp = model.generate_content(full_prompt, generation_config={"response_mime_type": "application/json"})
                raw = clean_json_response(resp.text)
                
            parsed = json.loads(raw)
            return {
                "id": idx,
                "text": text,
                "pred_category": parsed.get("category", "Unknown"),
                "pred_sub_category": parsed.get("sub_category", "Unknown")
            }
        except Exception as e:
            return {
                "id": idx, 
                "text": text, 
                "pred_category": "Error", 
                "pred_sub_category": "Error"
            }

    # 2. Parallel Execution
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(classify_single, row['text'], idx) for idx, row in sample_df.iterrows()]
        
        for future in as_completed(futures):
            results.append(future.result())
            
    return pd.DataFrame(results)