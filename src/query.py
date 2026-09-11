from embedder import Embedder
from db import DBConnection


def test_search(
    query_text,
    manufacturer="BMW",
    model="735i M Sport",
    model_year=None,
):
    print(f"Query: '{query_text}'")
    print(f"Targeting: {manufacturer} / {model} / {model_year}")

    embedder = Embedder()
    db = DBConnection()

    query_embedding = embedder.embed(query_text)

    results = db.hybrid_search(
        query=query_text,
        query_embedding=query_embedding,
        target_manufacturer=manufacturer,
        target_model=model,
        target_model_year=model_year,
        top_k=5,
    )

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(result)


if __name__ == "__main__":
    test_search(
        query_text="What audio system and amplifier output are specified for the 2024 BMW 735i M Sport?",
        model_year="2024",
    )