# Zomato CSAO Recommendation System: Evaluation Framework

This document outlines the evaluation strategy utilized for the Zomato CSAO Hackathon. To ensure the LightGBM LambdaMART ranker is robust and strictly aligned with the hackathon's Success Criteria, we have separated our evaluation into two distinct phases: Offline Evaluation and Online A/B Testing.

## 1. Offline Evaluation Strategy

Our offline evaluation rigorously tests the LightGBM model's ability to rank items accurately before deployment. We explicitly model this as a Learning-to-Rank (LTR) problem rather than simple binary classification.

### 1.1 Temporal Train-Test Split
As requested by the grading rubric, we **do not** use a random data split, as randomly splitting order history leaks future purchasing behavior into the training set. 
- **Historical Train Data (`data/historical_train_data.csv`)**: Comprises the older 80% of generated order sequences. This is used exclusively to train the LightGBM model to understand foundational item affinities and user tracking behavior over time.
- **Recent Test Data (`data/recent_test_data.csv`)**: Comprises the most recent 20% of orders. The model is completely blind to this dataset during training. This simulates strictly *future* inference requests.

### 1.2 Monitored Metrics
Our `evaluation/metrics.py` framework measures the test dataset against the following Information Retrieval (IR) metrics:
*   **AUC (Area Under ROC Curve):** Measures overall model discrimination ability (Target > 0.70).
*   **HitRate / Recall@K:** The percentage of times the *actual* user-added item appears within the system's generated top-K display rail.
*   **Precision@K:** The accuracy density of the top-K recommendations.
*   **NDCG (Normalized Discounted Cumulative Gain):** Validates the strict *order* of the list. An item ranked at position 1 is mathematically rewarded much more than an item ranked at position 3.
*   **MRR (Mean Reciprocal Rank):** Determines on average how high up the list the true relevant item appeared.

---

## 2. Online A/B Testing Strategy (Post-Development)

Once the model proves performant offline, the system must be deployed and validated live.

### Phase 1: A/A Testing (Sanity Check)
- Deploy the new Two-Stage Vector+LightGBM system shadowing the current baseline Zomato recommendation system.
- Log predictions silently to ensure the **Inference Latency** remains strictly below the 300ms SLA without affecting actual users.

### Phase 2: Canary and A/B Release
- **Control Group (A):** Users see the standard, static Zomato recommendation rail.
- **Variant Group (B):** Users receive predictions strictly from the LightGBM LambdaMART ranker.

### 2.1 Live Business Metrics to Monitor
The following metrics correspond directly to the targeted Business Impact success criteria:
*   **Add-on Acceptance Rate:** What percentage of users in Group B click 'Add' on our recommended side dishes compared to Group A?
*   **AOV Lift (Average Order Value):** Does the inclusion of hyper-personalized intelligent recommendations definitively increase the final checkout price of Group B's carts?
*   **C2O (Cart-to-Order) Ratio:** We must ensure that the new widget is non-intrusive. If Group B's cart abandonment rate rises, the widget friction outweighs the Add-on value.

### 2.2 Live Operational Metrics to Track
*   **Real-time Latency (P95/P99):** Ensure vector embeddings and LTR inference maintain < 300ms SLA at peak Zomato traffic scale.
*   **Catalog Coverage:** Monitor the percentage of total unique items recommended across all users to ensure the model exhibits proper diversity and is not simply recommending "Coke" and "Water" to every user.
