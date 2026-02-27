# Zomato CSAO Recommendation Engine

Welcome to the technical documentation for our Zomato Cross-Selling and Add-on Optimization (CSAO) engine. This system is designed to predict and recommend the optimal contextual add-ons (sides, desserts, beverages) to a user's cart in real-time, maximizing both cart value and user experience.

## The AI Edge (Differentiator)

Our architecture explicitly leverages modern Foundational Large Language Models (LLMs) to natively solve the unstructured data problem that plagues traditional Recommendation Systems.

Instead of relying on rigid, pre-programmed Database foreign-keys or fragile exact-string matching to map complementary items, our engine passes the cart contents through **`all-MiniLM-L6-v2` (a Sentence Transformer LLM)**. 

This model generates high-dimensional mathematical embeddings of the food items. It understands that "Pasta" and "Spaghetti" are semantically identical concepts without us ever programming a rule for it. By mean-pooling these embeddings, we can capture the "vibe" or context of complex, multi-item carts dynamically, calculating Cosine Similarity to find the absolute perfect cultural add-on in milliseconds. 

This "AI Edge" allows our pipeline to be entirely zero-shot and instantly scale to new, unstructured menu items or restaurants without requiring a single manual database update.

## File Map

Look in this directory for explicit documentation:
- `system_architecture_and_design.txt` (Breakdown of the Two-Stage LightGBM architecture)
- `scalability_considerations.txt` (Latency, SLA, and Benchmarking Strategy)
- `trade_offs_and_limitations.txt` (Offline vs Online metrics and margin constraints)
- `operational_metrics.txt` (Post-Execution pipeline latency numbers)
