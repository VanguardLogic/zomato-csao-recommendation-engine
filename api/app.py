import sys
import os

# Add required paths to sys.path so we can import modules
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "1_Model_Development"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

# Import inference engine
from online_api.inference import TwoStageEngine

# Initialize FastAPI app
app = FastAPI(title="Zomato CSAO Recommendation API")

# Load model engine eagerly at startup to ensure P99 < 200ms latency
print("Loading Recommendation Engine into memory...")
engine = TwoStageEngine()
print("Engine loaded successfully!")

class RecommendationRequest(BaseModel):
    cart_items: List[str]
    user_segment: Optional[str] = "Premium"
    time_of_day: Optional[str] = "Auto"

templates = Jinja2Templates(directory=os.path.join(base_dir, "api", "templates"))
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommend")
async def recommend(request: RecommendationRequest):
    try:
        # Detect Time of Day automatically if requested
        if request.time_of_day == "Auto" or not request.time_of_day:
            hour = datetime.now().hour
            if hour < 17: time_of_day = "Lunch"
            else: time_of_day = "Dinner"
        else:
            time_of_day = request.time_of_day
            
        user_segment = request.user_segment or "Premium"
        
        # We pre-loaded the engine, so inference should just be standard forward passes
        # <200ms target should easily be met
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
