import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_engine import core

# --- DATA MODELS ---
class AnalyzeRequest(BaseModel):
    text: str
    previous_text: Optional[str] = None  # User's previous message (for logic check)
    topic: str = "Artificial Intelligence" # The Debate Topic (for global check)

class EvidenceItem(BaseModel):
    text: str
    source: str
    url: str
    similarity: float  

class AnalyzeResponse(BaseModel):
    # Factual Results
    verdict: str
    factual_score: float
    explanation: str
    
    # Relevance Results
    relevance_score: float
    discourse_category: str
    discourse_reason: str
    
    # Debug/Math Details
    support_mass: float
    refute_mass: float
    
    evidence: List[EvidenceItem]

# --- LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 API Starting...")
    try:
        core.load_brain()
        print("✅ Brain ready!")
    except Exception as e:
        print(f"❌ Brain failed to load: {e}")
    yield
    print("💤 API Stopping...")

app = FastAPI(title="Debate Analyzer AI", lifespan=lifespan)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_claim(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Call the Orchestrator in core.py
        result = core.orchestrate_analysis(
            text=request.text,
            previous_text=request.previous_text,
            topic=request.topic
        )
        
        # Map Dict to Pydantic Model
        return AnalyzeResponse(
            verdict=result['fact_verdict'],
            factual_score=result['fact_score'],
            explanation=result['fact_explanation'],
            
            relevance_score=result['relevance_score'],
            discourse_category=result['relevance_category'],
            discourse_reason=result['relevance_reason'],
            
            support_mass=result['support_mass'],
            refute_mass=result['refute_mass'],
            
            evidence=[
                EvidenceItem(
                    text=f['text'],
                    source=f['source'],
                    url=f['url'],
                    similarity=f['similarity']
                ) for f in result['evidence']
            ]
        )
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))