# Zomato CSAO Recommendation Engine: Comprehensive Master Documentation

## 1. Executive Summary

The Zomato Cross-Selling & Add-on Optimization (CSAO) recommendation engine is a highly optimized, culturally aware AI inference pipeline designed to intelligently suggest complementary food items. Built specifically to handle Zomato's unique regional requirements (e.g., distinguishing North Indian from Italian cuisines, and filtering strictly by vegetarian constraints dynamically), the system replaces traditional, rigid "if/else" logic tags with semantic embeddings.

It solves the "Generic Fallback" problem (e.g. recommending Coke with everything, or a Margherita Pizza with Butter Chicken) by implementing an incredibly fast, highly tunable **Two-Stage Machine Learning Architecture**.

---

## 2. System Architecture and Core Design

The recommendation engine utilizes a robust **Two-Stage Architecture**, a pattern commonly found in enterprise-scale recommendation systems like YouTube and Pinterest. This funnel-based approach ensures deep accuracy without sacrificing P99 latency.

### Stage 1: Candidate Retrieval (Vector Semantic Search)
Running complex Machine Learning models against a massive regional catalog of thousands or millions of items is computationally impossible within a 300ms latency budget. Stage 1 acts as a strict "funnel."

1.  **Transformer Embeddings:** We use a pretrained transformer model (`sentence-transformers/all-MiniLM-L6-v2`) to encode the textual descriptions and metadata of the user's cart items into a dense, 384-dimensional vector space.
2.  **Context Vector Construction:** Instead of just examining the last item added to the cart, we use **Weighted Sequential Pooling**. The most recently added item carries 50% of the pooling weight, while the mean of all previous items in the cart accounts for the remaining 50%. This generates a single, highly accurate "Cart Context Vector."
3.  **Strict Cuisine Filtering (Stage 0):** Before any math is done, the system heuristically identifies the "Dominant Cuisine" of the cart. It strips out entirely unrelated categories (e.g., no South Indian food is searched if the cart is 100% Italian). Global items (Beverages/Desserts) remain valid.
4.  **Cosine Similarity Sweep:** We perform a rapidly indexed Cosine Similarity search between the Cart Context Vector and the pre-computed embeddings of our allowable dishes.
5.  **Output:** This retrieves an initially un-sorted subset of 50 highly contextually relevant "candidates."

### Stage 2: Machine Learning Ranker (LightGBM LambdaMART)
The 50 candidates retrieved from Stage 1 are passed through a dedicated Machine Learning model to determine the absolute optimal sort order for the user interface rail.

1.  **Feature Injection:** The model ingests a matrix of 11 features per candidate. This matrix includes generic context (Time of Day, User Segment based on historical activity), Item Features (Price, Global Popularity, Vegetarian constraints), and the Vector Affinity score from Stage 1. 
    > *Note: This is primarily a **Cart-Context Recommender**, relying on lightweight heuristics like `user_historical_veg_ratio` rather than monolithic sequence embeddings, allowing for lightning-fast cold starts.*
2.  **Pairwise Ranking (LambdaMART):** Using the LightGBM LambdaMART algorithm, the candidates are scored in pairs to maximize the probability of user acceptance based on historic interaction logs.
3.  **Diversity & Penalty Constraints:** Finally, the model outputs a probability score for each item. We apply a **Popularity Penalty (alpha = -0.1)** to globally popular items (like "Water") to actively discover unique, high-margin, culturally accurate pairings. A final Diversity Constraint ensures no more than 2 items of the same sub-category (e.g. Beverages) populate the final Top 8 recommendations returned to the UI.

### Architectural Trade-offs
- **Why LightGBM?** It natively supports the `lambdarank` algorithmic objective, mathematically the optimal way to solve "List Sorting" (Learning-to-Rank) problems compared to binary classification in XGBoost.
- **In-Memory vs Database:** For this MVP, embeddings and cosines are swept in-memory via JSON arrays and dictionaries. While lightning-fast for ~300 items, at scale, this trades deployment velocity for eventual memory bottlenecks.

---

## 3. Evaluation Framework

To ensure the LightGBM LambdaMART ranker is robust and strictly aligned with business goals, we separated evaluation into distinct Offline Evaluation and (hypothetical) Online A/B Testing phases.

### Offline Evaluation Strategy
We explicitly model this as a Learning-to-Rank (LTR) problem rather than simple binary classification.

- **Temporal Train-Test Split:** We **do not** use random `train_test_split`. Random splitting leaks future purchasing behavior into the training set. Instead, training is restricted to the older 80% of generated order sequences (`data/historical_train_data.csv`). The model is evaluated blind on the most recent 20% (`data/recent_test_data.csv`) to truly simulate future inference requests.
- **Monitored IR (Information Retrieval) Metrics:**
  - **AUC (Area Under ROC Curve):** Target > 0.70. Measures discrimination ability.
  - **HitRate / Recall@K:** The percentage of times the *actual* user-added item appears within the generated top-K display rail.
  - **Precision@K:** Accuracy density of the top-K.
  - **NDCG (Normalized Discounted Cumulative Gain):** Validates the strict order. A correct item at position 1 mathematically rewards the model more than catching it at position 3.
  - **MRR (Mean Reciprocal Rank):** Calculates how high up the list the relevant item appeared on average.

### Online A/B Testing Strategy (Post-MVP)
- **Phase 1 (Shadow Testing):** Deploy the new API to shadow the baseline system, logging predictions silently to verify live Inference Latency remains below 300ms without risking cart abandonment.
- **Phase 2 (Canary Release):** A true A/B split. 
  - *Group A:* Baseline static recommendation rail.
  - *Group B:* LightGBM LambdaMART predictions.
- **Live Business KPIs:**
  - **Add-on Acceptance Rate:** Click-through rate of Group B vs Group A.
  - **AOV Lift (Average Order Value):** Total checkout price delta.
  - **C2O (Cart-to-Order) Ratio:** Ensures the widget is non-intrusive (measuring cart abandonment).

---

## 4. Scalability Considerations

To ensure the engine can organically handle millions of daily prediction requests at Zomato's peak volume, we prioritized the following:

### Latency Optimization (< 300ms SLA)
- **Measured Inference:** The offline evaluation pipeline clocked an average end-to-end inference latency of **~40ms per request**, comfortably within the stringent SLA.
- **Precomputed Stage 1:** To guarantee this speed, **all catalog item embeddings are precomputed offline**. At runtime, we only invoke the transformer on the user's active cart items.

### Model Sizing
- **`all-MiniLM-L6-v2`:** Chosen for its incredibly lightweight footprint (~90 MB) while remaining perfectly adequate for semantic mapping of food terminology. Memory loading issues on containerized Kubernetes worker nodes are virtually non-existent.
- **LightGBM:** Tree storage structure using integer-mapped leaves rather than bloated floating-point memory heavily minimizes cold-start spin-up times for the API endpoints.

### Enterprise Infrastructure Scaling (The Final Evolution)
The current MVP logic runs fully natively pythonic logic. Transitioning to Zomato Scale requires upgrading the datastores:

1.  **Vector Databases:** The Stage-1 retrieval sweep (Cos. Sim dict) must transition into an **Approximate Nearest Neighbors (ANN)** clustering database (e.g., **FAISS, Milvus, Pinecone**). This enables Stage-1 to geometrically partition 1,000,000+ items and return the 50 candidates in < 10ms without a linear memory sweep.
2.  **Centralized Feature Stores:** Calculating rolling window features (e.g., `user_historical_veg_ratio`) on the fly represents unnecessary block-time during a 300ms SLA window. In enterprise production, these features are asynchronously computed nightly by Spark pipelines and loaded into an in-memory **Redis Feature Store**. The live endpoint simply pulls `redis.get(user_id)` instantly when constructing the LightGBM Stage 2 input array.
3.  **Data Ingestion:** Synthetic local CSV handling gives way to streaming batch ingestion directly from Zomato's **Snowflake** or **AWS S3 Data Lakes**, trained securely using scheduled **Apache Airflow** batch jobs.

---

## 5. Known Limitations
- **Synthetic Data Inflated Metrics:** Our offline evaluation trained on highly specific synthetic probability functions representing our assumptions of Zomato traffic. The model's final metrics (AUC ≈ 0.93, HitRate@8 ≈ 99%) represent its mastery over these tight synthetic patterns. In a production scenario with "noisy" real human data, a HitRate@10 ≈ 40-70% is standard and expected.
- **No Deep Personalization Profile:** Bypassing heavy Sequential User Embeddings prioritizes speed. Long-form user behavior is flattened into proxy heuristics, meaning the system may struggle to recommend deep paradigm shifts (e.g., a historically 100% vegetarian user suddenly searching for Chicken without obvious contextual clues).

---
*Built for the Zomato CSAO Hackathon. Refer to individual files within `3_Documentation/` for historical granular drafts.*
