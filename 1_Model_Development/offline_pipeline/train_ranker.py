# src/offline_pipeline/train_ranker.py

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import json
import lightgbm as lgb

def train_model():
    print("DEBUG: Loading training data...")
    df = pd.read_csv("data/synthetic_orders.csv")
    
    # Feature Engineering
    le_segment = LabelEncoder()
    le_freq = LabelEncoder()
    le_time = LabelEncoder()
    le_region = LabelEncoder()
    le_item = LabelEncoder()
    le_cart = LabelEncoder()
    
    with open("data/regional_affinity_map.json", "r") as f:
        graph = json.load(f)
        
    def get_embedding_score(row):
        cart = row['cart_items']
        cand = row['candidate_item']
        if cart in graph and cand in graph[cart]['candidates']:
            return graph[cart]['candidates'][cand]
        return 0.1
        
    df['embedding_affinity_score'] = df.apply(get_embedding_score, axis=1)
    
    # Sort and group for Learning-to-Rank algorithms (LambdaMART)
    df = df.sort_values('order_id')
    groups = df.groupby('order_id').size().values
    
    X = pd.DataFrame()
    X['user_segment'] = le_segment.fit_transform(df['user_segment'])
    X['order_frequency'] = le_freq.fit_transform(df['order_frequency'])
    X['time_of_day'] = le_time.fit_transform(df['time_of_day'])
    X['region'] = le_region.fit_transform(df['region'])
    X['candidate_item'] = le_item.fit_transform(df['candidate_item'])
    X['cart_items'] = le_cart.fit_transform(df['cart_items'])
    
    # Direct Numeric Features
    X['cart_total_value'] = df['cart_total_value']
    X['addon_price'] = df['addon_price']
    X['is_veg'] = df['is_veg']
    X['user_historical_veg_ratio'] = df['user_historical_veg_ratio']
    X['embedding_affinity_score'] = df['embedding_affinity_score']
    
    y = df['added']
    
    print("DEBUG: Training Stage 2 Ranker (LightGBM LambdaMART)...")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,        # More trees
        learning_rate=0.03,      # Slower learning
        num_leaves=31,           # Optimized leaf structure
        max_depth=10,            # Constrained depth to prevent overfitting
        min_child_samples=20,    # Regularization
        subsample=0.8,           # Row sampling
        colsample_bytree=0.8,    # Feature sampling
        random_state=42
    )
    model.fit(X, y, group=groups)
    
    # Save Model and Encoders
    artifacts = {
        "model": model,
        "encoders": {
            "segment": le_segment,
            "freq": le_freq,
            "time": le_time,
            "region": le_region,
            "item": le_item,
            "cart": le_cart
        }
    }
    
    with open("data/ranker_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("SUCCESS: Ranker model saved to data/ranker_model.pkl")

if __name__ == "__main__":
    train_model()
