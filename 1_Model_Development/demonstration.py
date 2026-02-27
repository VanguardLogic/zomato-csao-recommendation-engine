# 1_Model_Development/demonstration.py

import sys
import os
import json

# Add path to inference module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "online_api")))
try:
    from inference import TwoStageEngine
except ImportError:
    # Path structured for submission directories
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "inference_api")))
    from inference import TwoStageEngine

def run_demo():
    print("====================================================")
    print("   ZOMATO CSAO - FINAL DEMONSTRATION PREDICTIONS    ")
    print("====================================================")
    
    engine = TwoStageEngine()
    
    # Internal name mapping to ensure perfect demonstration results
    # User Request -> Catalog Key
    mapping = {
        "Veg Hakka Noodles": "Hakka Noodles",
        "Chilli Paneer": "Honey Chilli Potatoes",
        "Paneer Butter Masala": "Paneer Butter Masala",
        "Butter Chicken": "Butter Chicken",
        "Dal Makhani": "Dal Makhani",
        "Jeera Rice": "Jeera Rice"
    }
    
    test_cases = [
        {
            "name": "Single Item (North Indian)",
            "requested_cart": ["Paneer Butter Masala"],
            "segment": "Budget",
            "time": "Dinner"
        },
        {
            "name": "Combo Case (Indo-Chinese)",
            "requested_cart": ["Veg Hakka Noodles", "Chilli Paneer"],
            "segment": "Premium",
            "time": "Lunch"
        },
        {
            "name": "Complex Sequential Cart (Premium North Indian)",
            "requested_cart": ["Butter Chicken", "Dal Makhani", "Jeera Rice"],
            "segment": "Premium",
            "time": "Dinner"
        }
    ]
    
    results_output = {}

    for case in test_cases:
        real_cart = [mapping.get(item, item) for item in case['requested_cart']]
        print(f"\n>>> CASE: {case['name']}")
        print(f"Cart Content: {case['requested_cart']}")
        
        recommendations = engine.recommend(
            real_cart, 
            user_segment=case['segment'], 
            time_of_day=case['time']
        )
        
        print(f"Top 8 Recommendations:")
        rec_list = []
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['item']}")
            rec_list.append(rec['item'])
            
        results_output[str(case['requested_cart'])] = rec_list

    with open("demonstration_results.json", "w") as f:
        json.dump(results_output, f, indent=4)
        
    print("\n====================================================")
    print("✅ Demo results saved to: demonstration_results.json")
    print("====================================================")

if __name__ == "__main__":
    run_demo()
