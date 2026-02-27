from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ensure local imports work
sys.path.append(os.getcwd())

from src.online_inference.meal_state_machine import MealStateMachine

# Initialize FastAPI app
app = FastAPI(title="Zomato CSAO God-Mode API")

# Add CORS support for pre-built frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load state machine once on startup
sm = MealStateMachine()

class RecommendationRequest(BaseModel):
    cart_items: List[str]

@app.get("/")
async def root():
    return {
        "message": "Zomato CSAO God-Mode API is running",
        "endpoints": {
            "POST /recommend": "Get recommendations based on dishes",
            "GET /health": "Check system status",
            "GET /docs": "Interactive Swagger UI"
        }
    }

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        results = sm.get_recommendations(request.cart_items)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "God-Mode O(1)"}

if __name__ == "__main__":
    import uvicorn
    # Using 8001 to avoid Windows port 8000 "already in use" ghosts
    uvicorn.run(app, host="127.0.0.1", port=8001)
