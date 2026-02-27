# How to Pitch: Handling Missing or Unmapped Dishes

When you are presenting your prototype to the Zomato judges, they will immediately ask: **"Your dataset only has ~35 dishes. What happens when a restaurant adds 10,000 new items?"**

As a hackathon team, your job is not to build a production system that handles 10M rows of live data in 24 hours. Your job is to show that a **scalable mathematical architecture** is governing the system.

Here is exactly how you answer this question to win points for scalability:

## The Golden Answer

**You say:**
> *"Right now, our offline Knowledge Graph is populated with a limited dataset for prototype demonstration purposes. However, the architecture is built to handle an infinite catalog size without manually mapping every new dish."*

**When they ask "How?":**
> *"In a production environment, we do not rely on exact keyword matching. We would deploy a lightweight **Semantic Vector Embedding model** (like FAISS or OpenAI-Embeddings) at the ingestion layer."*

### Step-by-Step Explanation for the Judges:
1. **The Embedding Step:** When a restaurant adds an unmapped dish (like *"Cheesy Matar Paneer Surprise"*), the system converts the dish's name into a vector embedding.
2. **The Similarity Match:** It instantly compares that new vector against the core items already in the Knowledge Graph. It mathematically identifies that *"Cheesy Matar Paneer Surprise"* is 92% semantically similar to *"Matar Paneer"*.
3. **The Inheritance:** Without a single line of manual code or mapping, the new dish automatically "inherits" the candidate pool of Matar Paneer (e.g., Garlic Naan, Lassi).
4. **The Ranking:** Those inherited candidates are shot through our Live Machine Learning Ranker, which accurately predicts the best addon based on the user's specific budget, time, and historical order frequency.

## What this proves to the judges:
By giving this answer, you prove to them that:
1. You understand the limitations of a hackathon dataset.
2. You understand the "Cold Start / Unseen Item Problem" in recommendation engines.
3. You have an exact, modern MLOps architectural plan (Vector Search + Semantic Similarity) to solve it in production at Zomato scale. 

**Demo Proof:**
If they force you to type in something crazy today (like `"Spaghetti"`), the system will intelligently trigger its `Global Fallback`. Instead of crashing, your ML Ranker will catch the unmapped item and instantly fall back to generic, universally safe Zomato add-ons (like Water and Soft Drinks) and rank them based on the user's budget! You can run `test_hard.py` to show them this exact behavior!
