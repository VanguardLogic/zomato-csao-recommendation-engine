# src/online_api/app.py

import sys
import os

# Ensure local imports work by adding the project root to sys.path
# This MUST happen before importing from 'src'
sys.path.append(os.getcwd())

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.online_api.inference import TwoStageEngine
import uvicorn

from datetime import datetime
import random

app = FastAPI(title="Zomato CSAO Two-Stage ML API")
engine = TwoStageEngine()

class RecommendationRequest(BaseModel):
    cart_items: List[str]

@app.get("/")
async def root():
    return {
        "message": "Zomato CSAO Two-Stage ML API (Autonomous Mode)",
        "endpoints": {
            "POST /recommend": "Predict complements based on cart items only",
            "GET /docs": "Interactive Swagger UI"
        }
    }

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        # Detect Time of Day automatically
        hour = datetime.now().hour
        if 5 <= hour < 11: time_of_day = "Breakfast"
        elif 11 <= hour < 17: time_of_day = "Lunch"
        else: time_of_day = "Dinner"
        
        # Default user segment to 'Premium' for hackathon demo (better diversity)
        # In production, this would be fetched from a user session/DB
        user_segment = "Premium" 
        
        results = engine.recommend(
            request.cart_items, 
            user_segment, 
            time_of_day
        )
        return {
            "cart": request.cart_items,
            "inferred_context": {
                "time_of_day": time_of_day,
                "user_segment": user_segment
            },
            "recommendations": results,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
