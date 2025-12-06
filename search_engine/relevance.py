import os
import numpy as np
import google.generativeai as genai
import json
from dotenv import load_dotenv

# Load Env (Independent loading for modularity)
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Config
EMBEDDING_MODEL = "models/text-embedding-004"
VERIFIER_MODEL = "gemini-2.5-pro"

def get_embedding(text):
    """Generates a single vector for relevance comparison."""
    if not text:
        return np.zeros(768)
        
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return np.array(result['embedding'])
    except Exception as e:
        print(f"⚠️ Embedding Error in Relevance: {e}")
        return np.zeros(768)

def cosine_similarity(vec_a, vec_b):
    """Math helper for vector similarity."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def check_discourse_logic(current_text, previous_text):
    """
    Uses LLM to classify the logical relationship between two arguments.
    Returns: (Category, Score, Reason)
    """
    if not previous_text:
        return "OPENING_STATEMENT", 1.0, "This is the first statement in the context."

    prompt = f"""
    TASK: Analyze the debate logic between the PREVIOUS_ARGUMENT and the CURRENT_RESPONSE.

    PREVIOUS_ARGUMENT: "{previous_text}"
    CURRENT_RESPONSE: "{current_text}"

    INSTRUCTIONS:
    Classify the CURRENT_RESPONSE into exactly one category:
    1. DIRECT_COUNTER (Score 1.0): Directly refutes, challenges, or offers a counter-point to the Previous argument.
    2. ELABORATION (Score 0.7): Agrees, expands, adds examples, or asks a relevant clarifying question.
    3. TANGENTIAL (Score 0.2): Mentions related keywords/topics but ignores the specific logic or point of the Previous argument.
    4. IRRELEVANT (Score 0.0): Completely unrelated (e.g., talking about food in an AI debate).

    OUTPUT JSON ONLY:
    {{
        "category": "DIRECT_COUNTER" | "ELABORATION" | "TANGENTIAL" | "IRRELEVANT",
        "reason": "<short explanation>"
    }}
    """

    model = genai.GenerativeModel(VERIFIER_MODEL)
    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        category = data.get("category", "TANGENTIAL")
        reason = data.get("reason", "No reason provided")
        
        # Map Category to Score
        weights = {
            "DIRECT_COUNTER": 1.0,
            "ELABORATION": 0.7,
            "TANGENTIAL": 0.2,
            "IRRELEVANT": 0.0
        }
        score = weights.get(category, 0.0)
        
        return category, score, reason

    except Exception as e:
        print(f"⚠️ Discourse Logic Error: {e}")
        return "ERROR", 0.0, "LLM failed to analyze logic."

def compute_relevance_score(current_text, previous_text, topic):
    """
    Main entry point for Relevance Engine.
    Combines Global Topic Similarity (30%) + Local Discourse Logic (70%).
    """
    print(f"🔗 RELEVANCE ENGINE: Analyzing '{current_text[:20]}...'")

    # 1. Global Topic Relevance (Vector Sim)
    vec_topic = get_embedding(topic)
    vec_current = get_embedding(current_text)
    topic_sim = cosine_similarity(vec_topic, vec_current)
    
    # 2. Local Discourse Logic (LLM Classifier)
    category, logic_score, reason = check_discourse_logic(current_text, previous_text)
    
    # 3. Weighted Aggregation
    # We give 70% weight to responding to the opponent, 30% to staying on the general topic
    final_score = (topic_sim * 0.3) + (logic_score * 0.7)
    
    return {
        "final_score": round(final_score * 100, 2), # 0-100
        "topic_similarity": round(topic_sim, 2),
        "discourse_category": category,
        "discourse_reason": reason
    }