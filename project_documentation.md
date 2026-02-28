# CSAO Recommendation Engine: Technical Documentation

## 1. Overview

The Cross-Selling & Add-on Optimization (CSAO) engine is an AI pipeline built to suggest add-on food items. Instead of using hardcoded rules, we used semantic embeddings to handle regional cuisines and dietary preferences dynamically.

It solves the "Generic Fallback" problem (e.g. recommending Coke with everything, or a Margherita Pizza with Butter Chicken) by implementing a Two-Stage Machine Learning Architecture.

---

## 2. System Architecture and Core Design

The recommendation engine utilizes a robust **Two-Stage Architecture**, a pattern commonly found in enterprise-scale recommendation systems like YouTube and Pinterest. This funnel-based approach ensures deep accuracy without sacrificing P99 latency.

### Stage 1: Candidate Retrieval (Vector Semantic Search)
Running complex Machine Learning models against a massive regional catalog of thousands or millions of items is computationally impossible within a 300ms latency budget. Stage 1 acts as a strict "funnel."

1.  **Embeddings:** We use a pretrained model (`sentence-transformers/all-MiniLM-L6-v2`) to turn cart items into dense, 384-dimensional vectors.
2.  **Context Vector:** We use Weighted Sequential Pooling on the cart. The newest item gets 50% weight, and the mean of older items gets 50%, creating a single Cart Context Vector.
3.  **Cuisine Filter (Stage 0):** We heuristically find the dominant cuisine of the cart to strip out totally unrelated items early on.
4.  **Vector Search:** We run a Cosine Similarity search between the Cart Context Vector and all allowable dishes.
5.  **Output:** This returns an unsorted subset of 50 relevant candidates.

### Stage 2: Machine Learning Ranker (LightGBM LambdaMART)
The 50 candidates retrieved from Stage 1 are passed through a dedicated Machine Learning model to determine the absolute optimal sort order for the user interface rail.

1.  **Feature Injection:** The model ingests a matrix of features per candidate. This matrix includes Item Features (Price, Global Popularity, Vegetarian constraints) and the Vector Affinity score from Stage 1. 
    > *Note: This is strictly a **Cart-Context Recommender**, relying entirely on the composition of the current cart rather than monolithic sequence embeddings, allowing for lightning-fast cold starts.*
2.  **Ranking (LambdaMART):** Using LightGBM LambdaMART, the candidates are scored in pairs to predict the likelihood a user will add them based on historical data.
3.  **Refinement:** We apply a Popularity Penalty (alpha = -0.1) to generic items like "Water" to surface better pairings. A basic diversity check ensures we don't just output 8 beverages.

### Trade-offs
- **Why LightGBM?** It natively supports the `lambdarank` objective, which is better for "List Sorting" problems than binary classification.
- **In-Memory vs Database:** For this project, embeddings and cosine similarity operations run in-memory via numpy arrays. It's fast for ~300 items, but production scale would require an external vector database.

---

## 3. Evaluation Framework

To ensure the LightGBM LambdaMART ranker is robust and strictly aligned with business goals, we separated evaluation into distinct Offline Evaluation and (hypothetical) Online A/B Testing phases.

### Offline Evaluation Strategy
We explicitly model this as a Learning-to-Rank (LTR) problem rather than simple binary classification.

- **Temporal Split:** We purposefully avoid random `train_test_split` because it leaks future behavior. Training uses the older 80% of generated orders (`data/historical_train_data.csv`), and validation runs on the newer 20% (`data/recent_test_data.csv`).
- **Metrics:**
  - **AUC:** Target > 0.70 to confirm general classification ability.
  - **HitRate / Recall@K:** How often the real added item shows up in the top K displayed.
  - **NDCG:** Validates the sort order. A correct item at position 1 mathematically rewards the model more than catching it at position 3.

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

## 4. Scaling Up

To handle massive request volume in production:

### Latency Optimization (< 300ms SLA)
- **Speed:** Our offline tests show end-to-end inference takes **~40ms**, well under the 300ms SLA target.
- **Precomputed Stage 1:** To maintain speed, catalog item embeddings are generated offline. Runtime only evaluates the active cart.

### Model Sizing
- **`all-MiniLM-L6-v2`:** Chosen for its incredibly lightweight footprint (~90 MB) while remaining perfectly adequate for semantic mapping of food terminology. Memory loading issues on containerized Kubernetes worker nodes are virtually non-existent.
- **LightGBM:** Tree storage structure using integer-mapped leaves rather than bloated floating-point memory heavily minimizes cold-start spin-up times for the API endpoints.

### Production Infrastructure
Moving from MVP to full scale requires infrastructure upgrades:

1.  **Vector Databases:** The dict-bound retrieval sweep needs an ANN database (FAISS, Milvus) to instantly search through millions of items.
2.  **Feature Stores:** Features should be computed by batch jobs and loaded into an in-memory Redis cluster for the live endpoint to pull on demand.
3.  **Data Ingestion:** Local CSV workflows would be replaced by streaming data directly from Snowflake or S3.

---

## 5. Known Limitations
- **Synthetic Data Metrics:** The offline model trained on synthetic data representing our assumptions of food delivery traffic patterns. The high metrics (AUC ≈ 0.93) represent mastering those specific synthetic formulas. Real human data will have more noise and lower HitRates.
- **Personalization Limits:** Since we avoided heavy sequential user profiles in favor of speed, the system might struggle if a user suddenly shifts their usual dietary habits without clear cart signals.

---
*Refer to individual files within `3_Documentation/` for older methodology drafts.*
