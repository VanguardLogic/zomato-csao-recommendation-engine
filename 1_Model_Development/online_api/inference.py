# src/online_api/inference.py

import pickle
import json
import pandas as pd
import time
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class TwoStageEngine:
    """
    Two-Stage Recommendation Engine: 
    Stage 1: Vector Retrieval (all-MiniLM-L6-v2) with strict cuisine filtering.
    Stage 2: LightGBM Ranking (LambdaMART).
    """
    def __init__(self):
        data_path = "data/"
        if not os.path.exists(data_path):
            data_path = "../../data/"
            
        with open(os.path.join(data_path, "regional_affinity_map.json"), "r") as f:
            self.graph = json.load(f)
        with open(os.path.join(data_path, "ranker_model.pkl"), "rb") as f:
            artifacts = pickle.load(f)
            self.model = artifacts['model']
            self.encoders = artifacts['encoders']
            
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            self.encoder = None

    def recommend(self, cart_items, user_segment="Budget", time_of_day="Lunch", user_veg_ratio=0.5):
        # Stage 0: Detect primary cuisine region
        cart_regions = [self.graph.get(item, {}).get("region", "Unknown") for item in cart_items]
        cart_regions = [r for r in cart_regions if r != "Unknown"]
        dominant_region = Counter(cart_regions).most_common(1)[0][0] if cart_regions else "North Indian"
        
        # Stage 1: Candidate Retrieval (Top 50)
        candidate_scores = {}
        if self.encoder:
            vectors = [self.encoder.encode([item], normalize_embeddings=True)[0] for item in cart_items]
            
            if len(vectors) > 1:
                last_vec = vectors[-1]
                others_mean = np.mean(vectors[:-1], axis=0)
                others_mean = others_mean / (np.linalg.norm(others_mean) + 1e-9)
                context_vector = (0.5 * last_vec + 0.5 * others_mean).reshape(1, -1)
            else:
                context_vector = vectors[0].reshape(1, -1)
            
            similarities = []
            for dish_name, data in self.graph.items():
                if dish_name in cart_items: continue
                
                # Strict Cuisine Filtering
                item_region = data.get("region", "Unknown")
                if item_region not in [dominant_region, "Desserts", "Beverages"]:
                    continue
                
                dish_vec = np.array(data["embedding"]).reshape(1, -1)
                sim = cosine_similarity(context_vector, dish_vec)[0][0]
                
                # Popularity Penalty
                pop = data.get("popularity", 0.0)
                adjusted_score = sim - 0.1 * pop
                similarities.append((dish_name, adjusted_score))
            
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:50]
            for cand, score in similarities:
                candidate_scores[cand] = score

        if not candidate_scores:
            candidate_scores = {"Coke": 0.1, "Water": 0.1, "Fries": 0.1}

        # Stage 2: LightGBM Ranking
        unique_candidates = list(candidate_scores.keys())
        features = []
        for cand in unique_candidates:
            seg_enc = self.encoders["segment"].transform([user_segment])[0]
            time_enc = self.encoders["time"].transform([time_of_day])[0]
            reg_enc = self.encoders["region"].transform([dominant_region])[0]
            
            try:
                item_enc = self.encoders["item"].transform([cand])[0]
            except:
                item_enc = 0 
            
            is_veg = 1
            non_veg_keywords = ["Chicken", "Mutton", "Fish", "Prawn", "Keema", "Meat", "Egg", "Pepperoni"]
            if any(kw.lower() in cand.lower() for kw in non_veg_keywords): is_veg = 0
            
            features.append({
                "user_segment": seg_enc,
                "order_frequency": 1,
                "time_of_day": time_enc,
                "region": reg_enc,
                "candidate_item": item_enc,
                "cart_items": 0, 
                "cart_total_value": 300,
                "addon_price": 50,
                "is_veg": is_veg,
                "user_historical_veg_ratio": user_veg_ratio,
                "embedding_affinity_score": candidate_scores[cand]
            })
        
        if not features: return []
        
        probs = self.model.predict(pd.DataFrame(features))
        ranked_results = []
        for i, cand in enumerate(unique_candidates):
            score = float(probs[i])
            if cand == "Mango Shake":
                score *= 0.95
            ranked_results.append({"item": cand, "score": score})
            
        ranked_results = sorted(ranked_results, key=lambda x: x['score'], reverse=True)
        
        # Diversity Constraint: Max 2 Beverages
        beverages = ["Water", "Coke", "Soda", "Lassi", "Juice", "Tea", "Coffee", "Shake", "Drink"]
        final_top_8 = []
        bev_count = 0
        for res in ranked_results:
            if len(final_top_8) >= 8: break
            is_bev = any(bev.lower() in res['item'].lower() for bev in beverages)
            if is_bev:
                if bev_count < 2:
                    final_top_8.append(res)
                    bev_count += 1
            else:
                final_top_8.append(res)
        return final_top_8
