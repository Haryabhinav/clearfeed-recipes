import json
import random
import pandas as pd
import google.generativeai as genai
import openai
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def clean_json_response(text):
    text = str(text)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

def generate_descriptions(api_key, provider, cluster_data):
    total_clusters = sum(len(clusters) for clusters in cluster_data.values())
    logger.info(f"Generating descriptions for {total_clusters} clusters using {provider}...")

    prompt_input = {}
    for intent, clusters in cluster_data.items():
        prompt_input[intent] = {}
        for cluster_name, examples in clusters.items():
            # Ensure examples are strings and truncated to save tokens/reduce noise
            prompt_input[intent][cluster_name] = [str(e).replace("\n", " ")[:200] for e in examples[:3]]

    # FIXED PROMPT: Enforces exact key matching and deeper analysis
    prompt = f"""
    You are a data taxonomy expert. Your task is to define support ticket clusters based on the provided examples.

    INPUT DATA (JSON):
    {json.dumps(prompt_input, indent=2)}

    INSTRUCTIONS:
    1. **Analyze:** Read the examples for each cluster carefully.
    2. **Define:** Write a specific 1-sentence definition that explains EXACTLY what issue is contained in this cluster.
    3. **Constraint:** You must use the **EXACT JSON KEYS** (Category and Sub-category names) provided in the INPUT DATA. Do not change capitalization or wording of the keys.
    
    OUTPUT FORMAT (Strict JSON):
    {{
      "Exact Category Name": {{
         "Exact Sub-category Name": "Specific definition distinguishing this from other clusters."
      }}
    }}
    """
    try:
        if provider == "OpenAI":
            logger.info("Calling OpenAI API for description generation...")
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"✅ Description generation completed successfully")
            return result
        else:
            logger.info("Calling Gemini API for description generation...")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            result = json.loads(clean_json_response(response.text))
            logger.info(f"✅ Description generation completed successfully")
            return result
    except Exception as e:
        logger.error(f"❌ Description generation failed: {e}")
        return {}

def generate_classification_prompt(df, api_key, provider="Gemini"):
    logger.info("Starting classification prompt generation...")
    if df is None or df.empty:
        logger.warning("No data available to generate prompts")
        return "No data available to generate prompts."

    structured_data = {}
    valid_df = df[
        (df['cluster_category'] != "Unclustered") &
        (~df['cluster_category'].str.startswith("General "))
    ]
    logger.info(f"Found {len(valid_df)} valid tickets for prompt generation")
    
    for intent in valid_df['intent'].unique():
        structured_data[intent] = {}
        intent_df = valid_df[valid_df['intent'] == intent]
        for cluster in intent_df['cluster_category'].unique():
            cluster_df = intent_df[intent_df['cluster_category'] == cluster]
            texts = cluster_df['text'].tolist()
            clean_texts = [t.replace("\n", " ")[:200] for t in texts]
            structured_data[intent][cluster] = clean_texts
            
    descriptions_map = generate_descriptions(api_key, provider, structured_data)
    
    # IMPROVEMENT: High-readability Markdown Header & Table of Contents
    md_output = f"# 🏷️ Classification Taxonomy\n"
    md_output += f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
    
    md_output += "## 📋 Table of Contents\n"
    for intent in structured_data.keys():
        display_intent = intent.replace("_", " ").title()
        md_output += f"- [{display_intent}](#{intent.lower().replace(' ', '-')})\n"
    md_output += "\n---\n\n"

    for intent, clusters in structured_data.items():
        if not clusters: continue
        display_intent = intent.replace("_", " ").title()
        
        # Section Header
        md_output += f"## {display_intent} <a name='{intent.lower().replace(' ', '-')}'></a>\n\n"
        
        for cluster_name, all_examples in clusters.items():
            # Robust lookup: try exact match, then try fallback to avoid empty definitions
            intent_desc = descriptions_map.get(intent, {})
            desc = intent_desc.get(cluster_name, "Issues related to this specific topic.")
            
            # Sampling logic
            pos_examples = random.sample(all_examples, min(len(all_examples), 5))
            
            sibling_examples = []
            for sibling_name, sibling_data in clusters.items():
                if sibling_name != cluster_name:
                    sibling_examples.extend(sibling_data)
            
            neg_examples = []
            if sibling_examples:
                neg_examples = random.sample(sibling_examples, min(len(sibling_examples), 3))
            
            # IMPROVEMENT: Visual Hierarchy with Blockquotes and Bullets
            md_output += f"### 🔹 {cluster_name}\n"
            md_output += f"> **Definition:** {desc}\n\n"
            
            md_output += "**✅ Positive Examples:**\n"
            for p in pos_examples:
                clean_p = p.replace("\n", " ")[:120].strip()
                md_output += f"- *\"{clean_p}...\"*\n"
            
            if neg_examples:
                md_output += "\n**❌ Negative Examples (Do NOT match):**\n"
                for n in neg_examples:
                    clean_n = n.replace("\n", " ")[:120].strip()
                    md_output += f"- *\"{clean_n}...\"*\n"
            
            md_output += "\n---\n" # Separator

    logger.info(f"✅ Classification prompt generation completed ({len(md_output)} characters)")
    return md_output

def build_inference_system_prompt(definitions_md):
    return f"""
You are an expert support ticket classifier. 
Your goal is to categorize user requests strictly based on the taxonomy provided.

---
# TAXONOMY & DEFINITIONS
{definitions_md}
---

# 🧠 CLASSIFICATION STRATEGY
1. **Keyword Matching**: Look for the specific keywords mentioned in the definitions.
2. **Exclusion**: If a request mentions a topic found in a "Negative Example", discard that category.
3. **Specificity**: Prefer specific sub-categories (e.g., "Login Issues") over generic ones (e.g., "General Account Issues").

# OUTPUT FORMAT
Return a valid JSON object:
{{
  "category": "Exact Category Name",
  "sub_category": "Exact Sub-category Name",
  "confidence": "High/Medium/Low",
}}
"""

def validate_prompt(df, definitions_md, api_key, provider="Gemini"):
    logger.info("Starting prompt validation...")
    system_prompt = build_inference_system_prompt(definitions_md)
    if len(df) > 20:
        sample_df = df.sample(20).copy()
    else:
        sample_df = df.copy()

    logger.info(f"Validating prompt on {len(sample_df)} sample tickets using {provider}...")

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

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(classify_single, row['text'], idx) for idx, row in sample_df.iterrows()]
        for future in as_completed(futures):
            results.append(future.result())

    logger.info(f"✅ Prompt validation completed ({len(results)} tickets classified)")
    return pd.DataFrame(results)