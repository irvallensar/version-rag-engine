import csv
import os
import re

from embedder import Embedder
from db import DBConnection
from sentence_transformers import CrossEncoder


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def run_evaluation(csv_path: str, top_k: int = 5):
    embedder = Embedder()
    db = DBConnection()

    print("Loading Cross-Encoder Reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    total_queries = 0
    successful_retrievals = 0
    reciprocal_ranks = []

    print(f"--- Starting BMW 7 Series Evaluation (Top-{top_k}) ---")

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_text = row["question"]
            target_substring = row["target_substring"]
            manufacturer = row["manufacturer"]
            model = row["model"] or None
            model_year = row["model_year"] or None

            total_queries += 1
            query_vector = embedder.embed(query_text)

            initial_results = db.hybrid_search(
                query=query_text,
                query_embedding=query_vector,
                target_manufacturer=manufacturer,
                target_model=model,
                target_model_year=model_year,
                top_k=20,
            )

            if not initial_results:
                reciprocal_ranks.append(0.0)
                print(f"❌ Fail | No retrieval | Query: {query_text}")
                continue

            cross_inp = [[query_text, result[1]] for result in initial_results]
            cross_scores = reranker.predict(cross_inp)

            reranked_results = [
                result
                for _, result in sorted(
                    zip(cross_scores, initial_results),
                    key=lambda x: x[0],
                    reverse=True,
                )
            ]
            final_top_k = reranked_results[:top_k]

            found_rank = 0
            for rank, (chunk_id, content, metadata, score) in enumerate(
                final_top_k, 1
            ):
                if normalize_whitespace(target_substring) in normalize_whitespace(content):
                    found_rank = rank
                    break

            if found_rank > 0:
                successful_retrievals += 1
                reciprocal_ranks.append(1.0 / found_rank)
                print(
                    f"✅ Pass | {model_year} | Rank: {found_rank} | "
                    f"Query: {query_text}"
                )
            else:
                reciprocal_ranks.append(0.0)
                print(f"❌ Fail | {model_year} | Query: {query_text}")
                print(f"   Target: '{target_substring}'")

    recall = (
        (successful_retrievals / total_queries) * 100
        if total_queries > 0
        else 0
    )
    mrr = sum(reciprocal_ranks) / total_queries if total_queries > 0 else 0

    print("\n--- Final Metrics ---")
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Context Recall@{top_k}: {recall:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_csv_path = os.path.join(current_dir, "..", "tests", "eval_data.csv")
    run_evaluation(csv_path=absolute_csv_path, top_k=5)
