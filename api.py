import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# --- 1. PATH FIX ---
# Allow importing from the search_engine folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_engine import core

# --- 2. DATA MODELS (The Contract) ---
# This is what we expect the JS App to send us
class AnalyzeRequest(BaseModel):
    text: str

# This is the structure of a single piece of evidence
class EvidenceItem(BaseModel):
    text: str
    source: str
    url: str

# This is what we will send back to Java
class AnalyzeResponse(BaseModel):
    verdict: str
    confidence: int
    explanation: str
    evidence: List[EvidenceItem]

# --- 3. LIFECYCLE MANAGER ---
# This runs once when the server starts (to load the heavy Brain)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 API Starting... Waking up the Brain...")
    try:
        core.load_brain()
        print("✅ Brain is ready!")
    except Exception as e:
        print(f"❌ Failed to load brain: {e}")
        # We don't exit here, so the health check can still report status
    yield
    print("💤 API Shutting down...")

# --- 4. THE APPLICATION ---
app = FastAPI(title="Debate Analyzer AI", lifespan=lifespan)

@app.get("/")
def health_check():
    """Simple check to see if server is alive."""
    return {
        "status": "running", 
        "brain_loaded": core.INDEX is not None
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_claim(request: AnalyzeRequest):
    """
    Main Endpoint:
    1. Receives a text claim.
    2. Searches vector DB.
    3. Asks Gemini for verification.
    4. Returns JSON result.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        print(f"📨 Received request: {request.text[:50]}...")
        
        # A. Retrieval (RAG)
        facts = core.search(request.text, k=3)
        
        # B. Verification (LLM)
        llm_result = core.verify_claim_with_llm(request.text, facts)
        
        # C. Response Formatting
        return AnalyzeResponse(
            verdict=llm_result.get("verdict", "ERROR"),
            confidence=llm_result.get("confidence", 0),
            explanation=llm_result.get("explanation", "Processing error"),
            evidence=[
                EvidenceItem(
                    text=f['text'],
                    source=f['source'],
                    url=f['url']
                ) for f in facts
            ]
        )
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
