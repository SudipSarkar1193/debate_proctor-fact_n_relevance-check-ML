import sys
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json

# For path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from config import TOPIC_REGISTRY

# Import the new Relevance Module
from search_engine import relevance

# --- 2. CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API_KEY not found in .env file!")
else:
    print(f"🔑 Using API Key: {GEMINI_API_KEY}...")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/text-embedding-004"
VERIFIER_MODEL = "models/gemini-2.0-flash"


# --- 3. BRAIN MANAGER (Multi-Tenant) ---
ACTIVE_BRAINS = {}

def get_brain_paths(topic):
    """Returns file paths for a specific topic."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(base_path, '..')
    data_dir = os.path.join(root_path, 'data', topic)
    
    return {
        "index": os.path.join(data_dir, 'vector_store.index'),
        "meta": os.path.join(data_dir, 'metadata.pkl')
    }

def load_brain(topic):
    """Loads a specific topic's brain into memory."""
    if topic in ACTIVE_BRAINS:
        return ACTIVE_BRAINS[topic]
    
    print(f"🧠 Loading Brain for topic: [{topic.upper()}]...")
    paths = get_brain_paths(topic)
    
    if not os.path.exists(paths['index']) or not os.path.exists(paths['meta']):
        raise FileNotFoundError(f"❌ Brain files not found for '{topic}'. Run build_index.py first.")
        
    try:
        index = faiss.read_index(paths['index'])
        with open(paths['meta'], 'rb') as f:
            metadata = pickle.load(f)
            
        ACTIVE_BRAINS[topic] = {"index": index, "metadata": metadata}
        print(f"✅ Brain Loaded! ({len(metadata)} memories)")
        return ACTIVE_BRAINS[topic]
        
    except Exception as e:
        raise RuntimeError(f"Failed to load brain for {topic}: {e}")

def search(topic, query, k=3):
    """Topic-aware search."""
    brain = load_brain(topic)
    index = brain["index"]
    metadata = brain["metadata"]
    
    print(f"🔍 Searching [{topic.upper()}]: '{query[:50]}...'")
    
    result = genai.embed_content(
        model=MODEL_NAME,
        content=query,
        task_type="retrieval_query" 
    )
    query_vec = np.array([result['embedding']]).astype('float32')
    
    distances, indices = index.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(metadata): continue 
        
        meta = metadata[idx]
        l2_distance = float(distances[0][i])
        similarity = max(0.0, min(1.0, 1 - (l2_distance / 2)))
        
        results.append({
            "text": meta['text'],
            "source": meta['title'],
            "url": meta['url'],
            "similarity": similarity
        })
        
    return results

def verify_claim_with_llm(claim, facts):
    """Fact-Checking Logic."""
    evidence_text = ""
    for i, f in enumerate(facts):
        evidence_text += f"EVIDENCE #{i+1} (Sim: {f['similarity']:.2f}):\nSource: {f['source']}\nText: {f['text']}\n\n"
        
    prompt = f"""
    SYSTEM: You are a strict debate judge. 
    USER CLAIM: "{claim}"
    
    {evidence_text}
    
    INSTRUCTIONS:
    1. Determine if the Evidence SUPPORTS, CONTRADICTS, or is UNRELATED.
    2. Provide a confidence score (0-100).
    3. Output JSON ONLY: {{ "verdict": "...", "confidence": int, "explanation": "..." }}
    """
    
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
            safety_settings=safety_settings
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"verdict": "ERROR", "confidence": 0, "explanation": "LLM Failure"}

def calculate_mathematical_score(llm_result, facts):
    """Math Scoring Logic."""
    verdict = llm_result.get("verdict", "NOT_VERIFIABLE")
    confidence = llm_result.get("confidence", 0) / 100.0
    
    w_v = 1.0 if verdict == "SUPPORTED" else 0.0 if verdict == "CONTRADICTED" else 0.5
    support_mass = 0.0
    refute_mass = 0.0
    
    print("\n🧮 MATH ENGINE:")
    for f in facts:
        sim = f['similarity']
        reliability = 0.9 
        e_i = sim * confidence * w_v * reliability
        r_i = sim * confidence * (1 - w_v) * reliability
        support_mass += e_i
        refute_mass += r_i
        print(f"   - Fact '{f['source']}': Sim={sim:.2f} -> Support={e_i:.2f}, Refute={r_i:.2f}")

    total_mass = support_mass + refute_mass + 1e-9
    if total_mass < 0.1:
        raw_score = 0.0
    else:
        raw_score = (support_mass - refute_mass) / (total_mass + epsilon)
    
    final_accuracy = (raw_score + 1) / 2 * 100
    
    return {
        "support_mass": round(support_mass, 4),
        "refute_mass": round(refute_mass, 4),
        "final_accuracy_score": round(final_accuracy, 2)
    }

# --- MASTER ORCHESTRATOR ---
def orchestrate_analysis(text, previous_text=None, topic="ai"):
    """
    Combines Fact Checking + Relevance Evaluation.
    """
    print(f"\n📢 ORCHESTRATOR: Analyzing for Topic [{topic.upper()}]...")
    
    # 1. Fact Check (Uses Topic Brain)
    facts = search(topic, text, k=3)
    llm_fact = verify_claim_with_llm(text, facts)
    math_fact = calculate_mathematical_score(llm_fact, facts)
    
    # 2. Relevance Check (Uses Relevance Module)
    # Note: Relevance doesn't need the Vector Brain, it uses the Topic String.
    relevance_res = relevance.compute_relevance_score(text, previous_text, topic)
    
    return {
        # Facts
        "fact_verdict": llm_fact.get("verdict", "UNKNOWN"),
        "fact_confidence": llm_fact.get("confidence", 0),
        "fact_explanation": llm_fact.get("explanation", ""),
        "fact_score": math_fact['final_accuracy_score'],
        
        # Relevance
        "relevance_score": relevance_res['final_score'],
        "relevance_category": relevance_res['discourse_category'],
        "relevance_reason": relevance_res['discourse_reason'],
        
        # Evidence
        "evidence": facts
    }

# --- TEST ---
if __name__ == "__main__":
    # Example Debate Flow
    topic = "aadhaar"
    prev_arg = "Aadhaar destroys privacy by centralizing data."
    curr_resp = "Actually, Aadhaar uses a federated database structure, so it is secure."
    
    result = orchestrate_analysis(curr_resp, prev_arg, topic)
    
    print("\n" + "="*30)
    print("🎯 FINAL REPORT:")
    print(f"Fact Score: {result['fact_score']}% ({result['fact_verdict']})")
    print(f"Relevance:  {result['relevance_score']}% ({result['relevance_category']})")
    print("="*30)