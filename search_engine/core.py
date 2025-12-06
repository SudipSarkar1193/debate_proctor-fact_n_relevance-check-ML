import sys
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
import json
import re

# --- 1. PATH FIX (Critical) ---
# This tells Python to look in the parent folder for .env and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv

# --- 2. CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API_KEY not found in .env file!")

genai.configure(api_key=GEMINI_API_KEY)

# Settings (Must match what you used in Phase 2)
MODEL_NAME = "models/text-embedding-004"
VERIFIER_MODEL = "gemini-2.5-pro" 

# Global Brain Variables
INDEX = None
METADATA = None

def load_brain():
    """Loads the FAISS index and Metadata from disk into RAM."""
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
    3. Returns the text facts.
    """
    if INDEX is None:
        load_brain()
        
    print(f"🔍 Searching for: '{query}'")
    
    # 1. Embed Query
    result = genai.embed_content(
        model=MODEL_NAME,
        content=query,
        task_type="retrieval_query" 
    )
    query_vec = np.array([result['embedding']]).astype('float32')
    
    # 2. Search Index
    distances, indices = INDEX.search(query_vec, k)
    
    # 3. Retrieve Text
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue # No match found
        
        # FAISS ID -> Metadata List Index
        meta = METADATA[idx]
        
        results.append({
            "text": meta['text'],
            "source": meta['title'],
            "url": meta['url']
        })
        
    return results

def verify_claim_with_llm(claim, facts):
    """
    Sends the Claim + Facts to Gemini to get a Verdict.
    """
    # Prepare the Evidence Block
    evidence_text = ""
    for i, f in enumerate(facts):
        evidence_text += f"EVIDENCE #{i+1}:\nSource: {f['source']}\nText: {f['text']}\n\n"
        
    prompt = f"""
    SYSTEM: You are a strict debate judge. 
    You must verify the user's CLAIM based ONLY on the provided EVIDENCE.
    
    USER CLAIM: "{claim}"
    
    {evidence_text}
    
    INSTRUCTIONS:
    1. Compare the Claim vs Evidence.
    2. Determine if the Evidence SUPPORTS, CONTRADICTED, or is UNRELATED to the claim.
    3. Provide a confidence score (0-100).
    4. Provide a strict verdict label: "SUPPORTED", "CONTRADICTED", or "NOT_VERIFIABLE".
    5. Output JSON ONLY.
    
    JSON FORMAT:
    {{
        "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_VERIFIABLE",
        "confidence": <int>,
        "explanation": "<short sentence citing specific evidence>"
    }}
    """
    
    model = genai.GenerativeModel(VERIFIER_MODEL)
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    
    try:
        return json.loads(response.text)
    except:
        return {"verdict": "ERROR", "confidence": 0, "explanation": "LLM JSON Error"}

# --- TESTING AREA (Run this file to test) ---
if __name__ == "__main__":
    # Test Run
    load_brain()
    
    # A statement
    user_queries = ["Generative AI models like Stable Diffusion work by learning statistical patterns from vast datasets of images to create new visual content. Due to copyright concerns regarding this training data, the European Union's AI Act has officially banned the use and development of all generative AI models within Europe as of 2024.","Prominent researchers from organizations like the Future of Life Institute have raised valid concerns that advanced AI systems could pose an existential threat to humanity if not aligned with human values. Because of this misalignment, there is now a unanimous scientific consensus that AI will inevitably cause human extinction by the year 2030."]
    
    
    for user_query in user_queries :
        # 1. Retrieval
        facts = search(user_query, k=3)
        
        print(f"\n📚 Found {len(facts)} relevant facts:")
        for f in facts:
            print(f"   - [{f['source']}]: {f['text'][:100]}...")
            
        # 2. Verification
        print("\n🤖 Asking Judge (Gemini)...")
        result = verify_claim_with_llm(user_query, facts)
        
        print("\n" + "="*30)
        print("📢 FINAL VERDICT:")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reason: {result['explanation']}")
        print("="*30)