import sys
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
from dotenv import load_dotenv

# Import the new modular engine
# Note: Ensure __init__.py exists in search_engine folder or sys path is correct
from search_engine import relevance 

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ API_KEY not found in .env file!")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/text-embedding-004"
VERIFIER_MODEL = "gemini-2.5-pro" 

# Global Brain
INDEX = None
METADATA = None

def load_brain():
    """Loads FAISS index and Metadata."""
    global INDEX, METADATA
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(base_path, '..')
    index_path = os.path.join(root_path, 'vector_store.index')
    meta_path = os.path.join(root_path, 'metadata.pkl')
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ Index not found at {index_path}")
        
    print("🧠 Loading Vector Brain...")
    INDEX = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        METADATA = pickle.load(f)
    print(f"✅ Brain Loaded! ({len(METADATA)} memories)")

def search(query, k=3):
    """RAG Retrieval Step"""
    if INDEX is None:
        load_brain()
        
    # Embed
    result = genai.embed_content(model=MODEL_NAME, content=query, task_type="retrieval_query")
    query_vec = np.array([result['embedding']]).astype('float32')
    
    # Search
    distances, indices = INDEX.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue 
        meta = METADATA[idx]
        # Convert L2 distance to Cosine Similarity approximation
        similarity = max(0.0, min(1.0, 1 - (float(distances[0][i]) / 2)))
        
        results.append({
            "id": meta['faiss_id'],
            "text": meta['text'],
            "source": meta['title'],
            "url": meta['url'],
            "similarity": similarity
        })
    return results

def verify_claim_with_llm(claim, facts):
    """Fact-Checking Step (Phase 1)"""
    evidence_text = ""
    for i, f in enumerate(facts):
        evidence_text += f"EVIDENCE #{i+1} (Sim: {f['similarity']:.2f}):\n{f['text']}\n\n"
        
    prompt = f"""
    SYSTEM: You are a strict debate fact-checker.
    USER CLAIM: "{claim}"
    
    {evidence_text}
    
    INSTRUCTIONS:
    1. Verdict: SUPPORTED | CONTRADICTED | NOT_VERIFIABLE
    2. Confidence: 0-100
    3. Explanation: Short sentence.
    
    OUTPUT JSON ONLY.
    """
    
    model = genai.GenerativeModel(VERIFIER_MODEL)
    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"LLM Fact Error: {e}")
        return {"verdict": "ERROR", "confidence": 0, "explanation": "LLM Failure"}

def calculate_mathematical_score(llm_result, facts):
    """Math Scoring Step (Phase 1)"""
    verdict = llm_result.get("verdict", "NOT_VERIFIABLE")
    confidence = llm_result.get("confidence", 0) / 100.0
    
    w_v = 1.0 if verdict == "SUPPORTED" else (0.0 if verdict == "CONTRADICTED" else 0.5)
    
    support_mass = 0.0
    refute_mass = 0.0
    
    for f in facts:
        # Ei = Sim * Conf * w_v * Reliability(0.9)
        e_i = f['similarity'] * confidence * w_v * 0.9
        # Ri = Sim * Conf * (1-w_v) * Reliability(0.9)
        r_i = f['similarity'] * confidence * (1 - w_v) * 0.9
        
        support_mass += e_i
        refute_mass += r_i
        
    total = support_mass + refute_mass + 1e-9
    if total < 0.1:
        raw_score = 0.0
    else:
        raw_score = (support_mass - refute_mass) / total
        
    final_accuracy = (raw_score + 1) / 2 * 100
    return {
        "final_accuracy_score": round(final_accuracy, 2),
        "support_mass": round(support_mass, 4),
        "refute_mass": round(refute_mass, 4)
    }

def orchestrate_analysis(text, previous_text=None, topic="General"):
    """
    MASTER FUNCTION: Orchestrates the entire analysis pipeline.
    Combines Fact Checking (Core) + Relevance Evaluation (Relevance Module).
    """
    print(f"\n📢 ORCHESTRATOR: Processing '{text[:30]}...'")
    
    # --- PHASE 1: FACTUALITY (The "Truth") ---
    facts = search(text, k=3)
    llm_fact_res = verify_claim_with_llm(text, facts)
    math_fact_res = calculate_mathematical_score(llm_fact_res, facts)
    
    # --- PHASE 2: RELEVANCE (The "Flow") ---
    # We delegate this entirely to the new module
    relevance_res = relevance.compute_relevance_score(text, previous_text, topic)
    
    # --- COMBINE RESULTS ---
    return {
        # Factual Data
        "fact_verdict": llm_fact_res.get("verdict", "UNKNOWN"),
        "fact_confidence": llm_fact_res.get("confidence", 0),
        "fact_explanation": llm_fact_res.get("explanation", ""),
        "fact_score": math_fact_res['final_accuracy_score'],
        "support_mass": math_fact_res['support_mass'],
        "refute_mass": math_fact_res['refute_mass'],
        
        # Relevance Data
        "relevance_score": relevance_res['final_score'],
        "relevance_category": relevance_res['discourse_category'],
        "relevance_reason": relevance_res['discourse_reason'],
        
        # Evidence Data
        "evidence": facts
    }