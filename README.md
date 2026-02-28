# 📖 Zomato CSAO: The Narrative Journey & Technical Deep Dive
## *Building a Culturally Intelligent Recommendation Engine*

Welcome to the **Zomato Cross-Selling & Add-on Optimization (CSAO)** project repository. This project demonstrates a production-grade, culturally anchored machine learning recommendation engine designed to intelligently suggest complementary food items (add-ons) to a user's cart. 

This README provides a comprehensive guide to understanding the engine's journey, how it works under the hood, navigating the repository layout, finding the evaluation metrics, and running the system yourself.

---

## 🌑 Prologue: The Recommendation Crisis
At the start of this project, we faced a classic "Generic Fallback" problem. With a small catalog (~50 items) and simple logic, the system suffered from:
1.  **Cuisine Hallucinations:** Recommending a MARGHERITA PIZZA for a user ordering BUTTER CHICKEN.
2.  **The "Fries & Coke" Trap:** Globally popular items overwhelmed specific, high-value pairings.
3.  **Low Semantic Density:** The model couldn't distinguish between subtle culinary relationships.

**Our Mission:** Build an engine that "understands" food, respects culture, and scales to enterprise demands—all under a 300ms latency budget.

---

## 🧠 How It Works (The Engine Explained)

To solve the crisis, we moved away from manual "if/else" tags and embraced **AI-driven Semantic Embeddings** and a massive **Two-Stage Machine Learning Pipeline**.

### 1. In Simple Terms (For Everyone)
Imagine you are at a restaurant. If you order "Butter Chicken," a good waiter shouldn't offer you a Slice of Pizza. They should offer you "Garlic Naan" or "Jeera Rice." 
Our AI acts like that expert waiter:
- **It understands the menu**: It groups items by cuisine (North Indian, Italian, Desserts) so it never mixes incompatible foods.
- **It reads the room**: It knows if it is Lunch or Dinner, and whether you are a Premium or Budget user, adjusting the suggestions accordingly.
- **It learns relationships**: It studies thousands of past orders to learn that "Momos" go well with "Manchow Soup." 
- **It narrows it down and ranks them**: First, it pulls a list of 50 items that make logical sense. Then, it meticulously ranks those 50 items to give you the absolute best 8 recommendations within milliseconds.

### 2. In Technical Terms (For Engineers & Data Scientists)
The recommendation system uses a **Two-Stage Recommendation Funnel Pipeline**:

*   **Stage 1: Candidate Retrieval (Vector Search)**
    *   We expanded the catalog from **50 to 300+ items** across 7 distinct cuisines to create dense semantic clusters.
    *   We use a robust Transformer model (`all-MiniLM-L6-v2`) to generate 384-dimensional dense semantic embeddings for every dish in the catalog.
    *   **Weighted Sequential Pooling** is applied to the user's cart items. The last item added carries 50% of the weight, and the mean of all previous items carries the other 50%. This creates a dynamic "Context Vector."
    *   **Strict Cuisine Filtering (Stage 0)** ensures that we only retrieve the top 50 candidates that share the *same dominant cuisine* as the cart, plus global items (Beverages/Desserts). We compute Cosine Similarity between the Cart Context Vector and all allowable dish vectors to fetch these 50 candidates.
*   **Stage 2: Candidate Ranking (LightGBM LambdaMART)**
    *   The 50 candidates are passed to a highly-tuned **LightGBM Ranker** model.
    *   The model evaluates 11 complex features: User Segment, Time of Day, Cart Total Value, Dish Popularity, Vegetarian Constraints, and Embedding Affinity Scores.
    *   It outputs a final probability score for each item, which is then passed through a **Diversity Constraint** (e.g., maximum 2 beverages allowed) to produce the final Top 8 recommendations.
*   **Final Polish:** A **Popularity Penalty** (`-0.1` alpha) is applied to global generic items (Water, Coke) to force the engine to discover unique, high-margin pairings (like Raita or Garlic Naan).

---

## 📂 Repository Layout Deep-Dive

This repository is meticulously structured for both data scientists and software engineers. All submission requirements are mapped correctly to the folders below.

*   `1_Model_Development/`
    *   **`data_prep/`**: Scripts for synthesizing and generating order histories natively.
    *   **`offline_pipeline/`**: The core ML pipeline scripts (`build_graph.py` to compile item embeddings, `train_ranker.py` to train the LightGBM model).
    *   **Hyperparameter Tuning:** Check `hyperparameter_tuning_approach.txt` for details on our LambdaMART approach.
*   `2_Evaluation_Results/`
    *   **Where the tests live.** This folder contains Python scripts that run blind evaluations and output all statistical performance data.
    *   **Highlights**: Look here for `model_performance_metrics.txt` (ROC-AUC, HitRate scores), `blind_test_metrics.txt` for generalization tests, and `comparison_with_baseline.txt`.
*   `3_Documentation/`
    *   **System Design & Architecture.** Text files explaining system design, evaluation frameworks, scalability constraints, and the overarching trade-offs.
*   `4_Business_Impact_Analysis/`
    *   **The Business Case.** Here you will find files detailing how this model drives Average Order Value (AOV), segment performance, and recommendations for deployment strategy.
*   `api/`
    *   **The Live Production API.** Contains `app.py`—a blazingly fast FastAPI wrapper that safely loads the models globally on startup into memory and serves a beautiful intuitive HTML frontend at `GET /` and the core inference endpoint at `POST /api/recommend` in < 200ms.
*   `data/`
    *   *(Auto-generated during training)* Contains the massive generated datasets (CSVs) and the serialized model artifacts (`ranker_model.pkl` and `regional_affinity_map.json`).

---

## 📈 Where to Find the Metrics

For judges and reviewers looking to validate our results, please check these specific files within the repository:

*   **Model Performance (HitRate, NDCG, AUC-ROC):** 
    👉 `2_Evaluation_Results/model_performance_metrics.txt`
*   **Blind Test Generalization (AUC, HitRate):** 
    👉 `2_Evaluation_Results/blind_test_metrics.txt`
*   **AOV Lift & Revenue Projections:** 
    👉 `4_Business_Impact_Analysis/projected_lift_and_acceptance.txt`
*   **Speed and Operational Latency Profiles:** 
    👉 `2_Evaluation_Results/operational_metrics.txt`
*   **Error Analysis & Baseline Comparisons:** 
    👉 `2_Evaluation_Results/error_analysis_and_insights.txt` and `2_Evaluation_Results/comparison_with_baseline.txt`

---

## 🚀 How to Run the Project

Follow these steps to generate data, train the model, and launch the live API interface on your local machine.

### Prerequisites
Make sure you have Python 3.9+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 1. Run the Full ML Training Pipeline
This single orchestration script handles directory setup, synthesizes 15,000+ orders, generates transformer embeddings, and trains the LambdaMART ranker from scratch.
```bash
python run_full_pipeline.py
```
*Depending on your hardware, this might take 2-5 minutes to complete.*

### 2. Start the Live Recommendation API
Once the pipeline has successfully produced the `.pkl` and `.json` artifacts in the `data/` folder, you can spin up the unified web API.
```bash
python api/app.py
```

### 3. Test It Out!
Open your web browser and navigate to:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

You will be greeted by a clean Zomato-themed UI. Type in a sample cart like `Butter Chicken, Garlic Naan` and watch the Two-Stage ML engine return culturally matched add-ons in less than 200 milliseconds!

---
*Built for the Zomato CSAO Hackathon.*
