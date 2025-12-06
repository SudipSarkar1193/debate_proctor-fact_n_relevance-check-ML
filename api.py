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

class EvidenceItem(BaseModel):
    text: str
    source: str
    url: str
    similarity: float  

class AnalyzeResponse(BaseModel):
    verdict: str
    llm_confidence: int
    explanation: str
    
    # New Math Fields
    factual_score: float
    support_mass: float
    refute_mass: float
    
    evidence: List[EvidenceItem]

# --- APP LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 API Starting...")
    try:
        core.load_brain()
        print("✅ Brain ready!")
    except:
        print("❌ Brain failed to load")
    yield
    print("💤 API Stopping...")

app = FastAPI(title="Debate Analyzer AI", lifespan=lifespan)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_claim(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        print(f"📨 Analyzing: {request.text[:50]}...")
        
        # 1. Search
        facts = core.search(request.text, k=3)
        
        # 2. LLM Verify
        llm_result = core.verify_claim_with_llm(request.text, facts)
        
        # 3. Math Calculate
        math_result = core.calculate_mathematical_score(llm_result, facts)
        
        # 4. Return Full Report
        return AnalyzeResponse(
            verdict=llm_result.get("verdict", "ERROR"),
            llm_confidence=llm_result.get("confidence", 0),
            explanation=llm_result.get("explanation", "Error"),
            
            # Math Data
            factual_score=math_result['final_accuracy_score'],
            support_mass=math_result['support_mass'],
            refute_mass=math_result['refute_mass'],
            
            evidence=[
                EvidenceItem(
                    text=f['text'],
                    source=f['source'],
                    url=f['url'],
                    similarity=f['similarity']
                ) for f in facts
            ]
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))