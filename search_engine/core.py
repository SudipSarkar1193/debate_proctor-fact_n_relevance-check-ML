import sys
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import re

# This tells Python to look in the parent folder for .env and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv

# --- 2. CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API_KEY not found in .env file!")

genai.configure(api_key=GEMINI_API_KEY)

# Settings
MODEL_NAME = "models/text-embedding-004"
VERIFIER_MODEL = "gemini-2.5-pro" 

# Global Brain Variables
INDEX = None
METADATA = None

def load_brain():
    """Loads the FAISS index and Metadata from disk into the RAM."""
    global INDEX, METADATA
    
    # Paths are relative to the ROOT folder (one level up)
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(base_path, '..')
    
    index_path = os.path.join(root_path, 'vector_store.index')
    meta_path = os.path.join(root_path, 'metadata.pkl')
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ Index not found at {index_path}! Run Phase 2 first.")
        
    print("🧠 Loading Vector Brain...")
    INDEX = faiss.read_index(index_path)
    
    with open(meta_path, 'rb') as f:
        METADATA = pickle.load(f)
    print(f"✅ Brain Loaded! ({len(METADATA)} memories available)")

def search(query, k=3):
    """
    1. Embeds the query.
    2. Searches FAISS for top K matches.
    3. Calculates Cosine Similarity from L2 Distance.
    """
    if INDEX is None:
        load_brain()
        
    print(f"🔍 Searching for: '{query[:50]}...'")
    
    # 1. Embed Query
    result = genai.embed_content(
        model=MODEL_NAME,
        content=query,
        task_type="retrieval_query" 
    )
    query_vec = np.array([result['embedding']]).astype('float32')
    
    # 2. Search Index
    distances, indices = INDEX.search(query_vec, k)
    
    # 3. Retrieve Text & Calculate Math
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        
        meta = METADATA[idx]
        l2_distance = float(distances[0][i])
        
        # MATH: Convert L2 Distance to Cosine Similarity
        # Formula: Similarity = 1 - (Distance^2 / 2)
        # Note: FAISS returns squared Euclidean distance already
        similarity = 1 - (l2_distance / 2)
        
        # Safety clamp (0.0 to 1.0)
        similarity = max(0.0, min(1.0, similarity))
        
        results.append({
            "id": meta['faiss_id'],
            "text": meta['text'],
            "source": meta['title'],
            "url": meta['url'],
            "similarity": similarity  # <--- CRITICAL FOR SCORING
        })
        
    return results

def verify_claim_with_llm(claim, facts):
    """
    Asks LLM to judge the claim based on facts.
    Includes Safety Settings to prevent blocking sensitive topics (e.g. Extinction).
    """
    evidence_text = ""
    for i, f in enumerate(facts):
        # We include similarity in the prompt so the LLM knows which evidence is strongest
        evidence_text += f"EVIDENCE #{i+1} (Relevance: {f['similarity']:.2f}):\nSource: {f['source']}\nText: {f['text']}\n\n"
        
    prompt = f"""
    SYSTEM: You are a strict debate judge. 
    Compare the USER CLAIM vs the EVIDENCE.
    
    USER CLAIM: "{claim}"
    
    {evidence_text}
    
    INSTRUCTIONS:
    1. Determine if the Evidence SUPPORTS, CONTRADICTED, or is UNRELATED to the claim.
    2. Provide a confidence score (0-100) for your verdict.
    3. Output JSON ONLY.
    
    JSON FORMAT:
    {{
        "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_VERIFIABLE",
        "confidence": <int>,
        "explanation": "<short sentence>"
    }}
    """
    
    # --- SAFETY SETTINGS (Disable filters for debate analysis) ---
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    model = genai.GenerativeModel(VERIFIER_MODEL)
    
    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"},
            safety_settings=safety_settings # <--- SAFETY APPLIED HERE
        )
        return json.loads(response.text)
        
    except Exception as e:
        print(f"LLM Error: {e}")
        # If it was a safety block, print the feedback to know for sure
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             print(f"Safety Feedback: {e.response.prompt_feedback}")
        return {"verdict": "ERROR", "confidence": 0, "explanation": "LLM Failure"}

def calculate_mathematical_score(llm_result, facts):
    """
    Calculates the Factual Accuracy Score (0-100) using Support vs Refute Mass.
    """
    
    # 1. Get LLM signals
    verdict = llm_result.get("verdict", "NOT_VERIFIABLE")
    confidence = llm_result.get("confidence", 0) / 100.0 # Normalize to 0-1
    
    # 2. Assign Verdict Weight (w_v)
    if verdict == "SUPPORTED":
        w_v = 1.0
    elif verdict == "CONTRADICTED":
        w_v = 0.0
    else:
        w_v = 0.5 # Neutral
        
    support_mass = 0.0
    refute_mass = 0.0
    
    # 3. Calculate Mass for every piece of evidence
    print("\n🧮 MATH ENGINE:")
    for f in facts:
        sim = f['similarity']
        # Reliability defaults to 0.9 for Wikipedia (could be variable in future)
        reliability = 0.9 
        
        # FORMULAS:
        # Support Mass (Ei) = Similarity * Confidence * w_v * Reliability
        e_i = sim * confidence * w_v * reliability
        
        # Refute Mass (Ri) = Similarity * Confidence * (1 - w_v) * Reliability
        r_i = sim * confidence * (1 - w_v) * reliability
        
        support_mass += e_i
        refute_mass += r_i
        
        print(f"   - Fact '{f['source']}': Sim={sim:.2f} -> Support={e_i:.2f}, Refute={r_i:.2f}")

    # 4. Final Aggregation
    total_mass = support_mass + refute_mass
    epsilon = 1e-9
    
    if total_mass < 0.1:
        # If total mass is tiny, we don't have enough evidence to score
        raw_score = 0.0
    else:
        # Score between -1 (Refuted) and +1 (Supported)
        raw_score = (support_mass - refute_mass) / (total_mass + epsilon)
    
    # Normalize to 0-100 scale
    final_accuracy = (raw_score + 1) / 2 * 100
    
    return {
        "support_mass": round(support_mass, 4),
        "refute_mass": round(refute_mass, 4),
        "raw_score": round(raw_score, 4),
        "final_accuracy_score": round(final_accuracy, 2)
    }

# --- TESTING AREA ---
if __name__ == "__main__":
    load_brain()
    
    # Test Claims
    user_queries = [
        "Prominent researchers have raised valid concerns that advanced AI systems could pose an existential threat to humanity if not aligned with human values."
    ]
    
    for query in user_queries:
        # 1. Retrieval
        facts = search(query)
        
        # 2. Verification
        llm_res = verify_claim_with_llm(query, facts)
        
        # 3. Math
        math_res = calculate_mathematical_score(llm_res, facts)
        
        print("\n" + "="*30)
        print("📊 FINAL CALCULATION:")
        print(f"Verdict: {llm_res['verdict']}")
        print(f"LLM Conf: {llm_res['confidence']}%")
        print("-" * 15)
        print(f"Support Mass: {math_res['support_mass']}")
        print(f"Refute Mass:  {math_res['refute_mass']}")
        print(f"Factual Acc:  {math_res['final_accuracy_score']}/100")
        print("="*30)