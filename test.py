from embedder import Embedder
from db import DBConnection

db = DBConnection()
embedder = Embedder()

test_queries = [
    ("What alloy wheels are fitted to the 2024 BMW 735i M Sport?", "735i M Sport", "2024"),
    ("What is the top speed listed for the 2024 BMW 735i M Sport?", "735i M Sport", "2024"),
    ("What is the engine displacement of the 2025 BMW 735i M Sport?", "735i M Sport", "2025"),
    ("What fuel type is required for the 2025 BMW 735i M Sport?", "735i M Sport", "2025"),
    ("What type of seats are in the rear of the 2025 BMW i7?", "i7 xDrive60", "2025"),
    ("What size are the light alloy wheels on the 2022 BMW 730Li M Sport?", "730Li M Sport", "2022"),
]

for query, model, year in test_queries:
    vec = embedder.embed(query)
    results = db.hybrid_search(
        query=query,
        query_embedding=vec,
        target_manufacturer="BMW",
        target_model=model,
        target_model_year=year,
        top_k=3,
    )
    print(f"\n====================\nQUERY: {query}")
    for idx, (_, content, meta, _) in enumerate(results, 1):
        print(f"--- Result {idx} ({meta.get('model')} {meta.get('model_year')}) ---")
        print(content[:300].replace("\n", " "))
