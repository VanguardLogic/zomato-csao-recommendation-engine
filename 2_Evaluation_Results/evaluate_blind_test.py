import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

# Ensure local imports work
sys.path.append(os.getcwd())

from importlib.machinery import SourceFileLoader
data_gen = SourceFileLoader("generate_synthetic_data", "1_Model_Development/data_prep/generate_synthetic_data.py").load_module()

def run_blind_evaluation():
    print("===============================================================")
    print("   ZOMATO CSAO - BLIND DATASET EVALUATION (THE UNSEEN TEST)  ")
    print("===============================================================")
    
    # 1. Generate an entirely new "blind" dataset the model has never seen
    print("DEBUG: Generating a completely new BLIND dataset of 3,000 orders...")
    # Temporarily override the save paths in data_gen if needed, but we can just use the returned dataframe
    df_blind = data_gen.generate_orders(num_orders=3000)
    
    # We treat this entire 3000 order set as the holdout
    positive_samples = df_blind[df_blind['added'] == 1]
    
    print("DEBUG: Loading the previously trained LightGBM LambdaMART ranker...")
    import pickle
    with open("data/ranker_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        encoders = artifacts['encoders']
        
    print("DEBUG: Scoring blind dataset...")
    
    # Safely transform candidates, dropping unseen items just like in production
    valid_mask = df_blind['candidate_item'].isin(encoders["item"].classes_) & df_blind['cart_items'].isin(encoders["cart"].classes_)
    df_valid = df_blind[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("CRITICAL: The blind dataset generated completely unseen items not in the encoders. Please ensure the master catalog matches.")
        return

    X_auc = pd.DataFrame()
    X_auc['user_segment'] = encoders["segment"].transform(df_valid['user_segment'])
    try:
        X_auc['order_frequency'] = encoders["freq"].transform(df_valid['order_frequency'])
    except:
        X_auc['order_frequency'] = 1 # Fallback
    X_auc['time_of_day'] = encoders["time"].transform(df_valid['time_of_day'])
    X_auc['region'] = encoders["region"].transform(df_valid['region'])
    X_auc['candidate_item'] = encoders["item"].transform(df_valid['candidate_item'])
    X_auc['cart_items'] = encoders["cart"].transform(df_valid['cart_items'])
    
    # Direct Numeric Features
    X_auc['cart_total_value'] = df_valid['cart_total_value'].values
    X_auc['addon_price'] = df_valid['addon_price'].values
    X_auc['is_veg'] = df_valid['is_veg'].values
    X_auc['user_historical_veg_ratio'] = df_valid['user_historical_veg_ratio'].values
    
    import json
    with open("data/regional_affinity_map.json", "r") as f:
        graph = json.load(f)
        
    def get_llm_score(row):
        cart = row['cart_items']
        cand = row['candidate_item']
        if cart in graph and cand in graph[cart]['candidates']:
            return graph[cart]['candidates'][cand]
        return 0.1
        
    X_auc['llm_affinity_score'] = df_valid.apply(get_llm_score, axis=1).values
    
    y_true = df_valid['added']
    y_prob = model.predict(X_auc)
    
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_score = 0.5 # Fallback if only one class exists in small sample
        
    # Hacky Ranking Eval (Without running the full TwoStageEngine logic which takes time)
    # Group by order_id
    df_valid['score'] = y_prob
    df_grouped = df_valid.groupby('order_id')
    
    mrr_sum = 0
    ndcg_sum = 0
    hits = 0
    valid_evals = 0
    K = 8 # Top 8 rail
    
    for order_id, group in df_grouped:
        # Sort group by predicted score descending
        sorted_group = group.sort_values(by='score', ascending=False).reset_index(drop=True)
        # Find the rank of the item that was actually added (added == 1)
        true_add_indices = sorted_group.index[sorted_group['added'] == 1].tolist()
        
        if not true_add_indices:
            continue
            
        valid_evals += 1
        best_rank = true_add_indices[0] + 1 # 1-indexed rank
        
        if best_rank <= K:
            hits += 1
        
        mrr_sum += 1.0 / best_rank
        ndcg_sum += 1.0 / np.log2(best_rank + 1)
        
    mrr = mrr_sum / valid_evals if valid_evals > 0 else 0
    ndcg = ndcg_sum / valid_evals if valid_evals > 0 else 0
    hit_rate = hits / valid_evals if valid_evals > 0 else 0
    
    print("\n--- BLIND EVALUATION METRICS (TRULY UNSEEN DATA) ---")
    print(f"Blind AUC         : {auc_score:.4f}")
    print(f"Blind HitRate @ {K} : {hit_rate:.2%}")
    print(f"Blind NDCG        : {ndcg:.4f}")
    print(f"Blind MRR         : {mrr:.4f}")
    print("===============================================================")
    
    print("\n--- SAMPLE BLIND PREDICTIONS ---")
    
    # Show predictions for the first 3 orders
    sample_count = 0
    for order_id, group in df_grouped:
        if sample_count >= 3:
            break
            
        cart_item = group['cart_items'].iloc[0]
        region = group['region'].iloc[0]
        time = group['time_of_day'].iloc[0]
        
        sorted_group = group.sort_values(by='score', ascending=False)
        top_k_items = sorted_group.head(8)
        actual_added = sorted_group[sorted_group['added'] == 1]
        
        print(f"\nOrder {order_id} [{region} | {time}]")
        print(f"Cart Contains : {cart_item}")
        
        if len(actual_added) > 0:
            print(f"User Added    : {actual_added['candidate_item'].iloc[0]} (Rank: {sorted_group.index.get_loc(actual_added.index[0]) + 1})")
        else:
            print(f"User Added    : [No Item]")
            
        print("Model Predicted:")
        for idx, (_, row) in enumerate(top_k_items.iterrows(), 1):
            print(f"  {idx}. {row['candidate_item']} (Score: {row['score']:.4f})")
            
        sample_count += 1

    
    with open("2_Evaluation_Results/blind_test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Blind AUC: {auc_score:.4f}\nBlind HitRate@{K}: {hit_rate:.2%}\nBlind NDCG: {ndcg:.4f}\nBlind MRR: {mrr:.4f}")
        
    print("Saved results to 2_Evaluation_Results/blind_test_metrics.txt")

if __name__ == "__main__":
    run_blind_evaluation()
