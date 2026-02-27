# src/offline_pipeline/build_graph.py

import json
import os
import pandas as pd
from collections import defaultdict

def build_graph():
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run `pip install sentence-transformers`")
        return

    print("DEBUG: Building regional affinity map with precomputed Semantic Embeddings for 300+ items...")
    
    # Load synthetic data to extract ALL unique catalog items and their co-occurrences
    try:
        df = pd.read_csv("data/synthetic_orders.csv")
    except:
        print("ERROR: data/synthetic_orders.csv not found. Run generate_synthetic_data.py first.")
        return
        
    all_items = set(df['cart_items'].unique()).union(set(df['candidate_item'].unique()))
    
    # Map item -> region for inference ranker consistency
    # We take the most frequent region associated with each item
    item_region_map = df.groupby('candidate_item')['region'].agg(lambda x: x.value_counts().index[0]).to_dict()
    # Also check cart_items
    cart_region_map = df.groupby('cart_items')['region'].agg(lambda x: x.value_counts().index[0]).to_dict()
    item_region_map.update(cart_region_map)
    
    # Compute popularity (occurrence frequency) to penalize generic items during inference
    item_counts = df['candidate_item'].value_counts().to_dict()
    total_candidates = len(df)
    # Normalize popularity to [0, 1] range
    max_count = max(item_counts.values()) if item_counts else 1
    popularity_map = {item: count / max_count for item, count in item_counts.items()}
    
    # Compute true candidates from data where added == 1
    success_df = df[df['added'] == 1]
    co_occurrences = defaultdict(lambda: defaultdict(int))
    
    for _, row in success_df.iterrows():
        cart = row['cart_items']
        cand = row['candidate_item']
        co_occurrences[cart][cand] += 1
        
    # Build the final graph structure
    affinity_map = {}
    for dish in all_items:
        # 1. Normalize embeddings as requested
        vector = encoder.encode([dish], normalize_embeddings=True)[0].tolist()
        
        # 2. Extract Top 15 specific candidates for exact match priors
        dish_candidates = co_occurrences[dish]
        total_co_occurrences = sum(dish_candidates.values())
        candidates_dict = {}
        if total_co_occurrences > 0:
            # Normalize scores
            candidates_dict = {k: v / total_co_occurrences for k, v in dish_candidates.items()}
            # Sort and take top 15
            candidates_dict = dict(sorted(candidates_dict.items(), key=lambda item: item[1], reverse=True)[:15])
            
        affinity_map[dish] = {
            "region": item_region_map.get(dish, "North Indian"), # Preserve true region!
            "embedding": vector,
            "popularity": popularity_map.get(dish, 0.0), # Normalized popularity
            "candidates": candidates_dict
        }
            
    os.makedirs("data", exist_ok=True)
    with open("data/regional_affinity_map.json", "w") as f:
        json.dump(affinity_map, f)
        
    print(f"SUCCESS: Knowledge Graph built with {len(all_items)} normalized Item Embeddings.")

if __name__ == "__main__":
    build_graph()
