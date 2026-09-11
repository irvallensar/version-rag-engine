import time
import statistics
import csv
import os
import re
from embedder import Embedder
from db import DBConnection
from sentence_transformers import CrossEncoder

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def run_extended_benchmark(csv_path: str, top_k: int = 5):
    embedder = Embedder()
    db = DBConnection()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    retrieval_latencies = []
    rerank_latencies = []
    total_latencies = []
    hit_ranks = []
    total_queries = 0
    hits = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["question"]
            target = row["target_substring"]
            mfg = row["manufacturer"]
            model = row["model"] or None
            year = row["model_year"] or None
            total_queries += 1

            # 1. Embedding + DB Search Latency
            t0 = time.perf_counter()
            vec = embedder.embed(query)
            initial_results = db.hybrid_search(
                query=query,
                query_embedding=vec,
                target_manufacturer=mfg,
                target_model=model,
                target_model_year=year,
                top_k=20,
            )
            t1 = time.perf_counter()
            retrieval_latencies.append((t1 - t0) * 1000)

            # 2. Reranker Latency
            cross_inp = [[query, res[1]] for res in initial_results]
            scores = reranker.predict(cross_inp)
            ranked = [
                res for _, res in sorted(
                    zip(scores, initial_results), key=lambda x: x[0], reverse=True
                )
            ][:top_k]
            t2 = time.perf_counter()
            rerank_latencies.append((t2 - t1) * 1000)
            total_latencies.append((t2 - t0) * 1000)

            # Hit rank tracking
            found = False
            for rank, (_, content, _, _) in enumerate(ranked, 1):
                if normalize_whitespace(target).lower() in normalize_whitespace(content).lower():
                    hit_ranks.append(rank)
                    hits += 1
                    found = True
                    break
            if not found:
                hit_ranks.append(0)

    # Computations
    hit_at_1 = sum(1 for r in hit_ranks if r == 1) / total_queries * 100
    hit_at_3 = sum(1 for r in hit_ranks if 1 <= r <= 3) / total_queries * 100
    recall_k = (hits / total_queries) * 100
    mrr = statistics.mean([1.0 / r if r > 0 else 0.0 for r in hit_ranks])

    p50_total = statistics.median(total_latencies)
    p95_total = sorted(total_latencies)[int(len(total_latencies) * 0.95)]
    avg_retrieval = statistics.mean(retrieval_latencies)
    avg_rerank = statistics.mean(rerank_latencies)

    print("\n================ BENCHMARK REPORT ================")
    print(f"Total Test Cases:            {total_queries}")
    print(f"Hit@1 Precision:             {hit_at_1:.2f}%")
    print(f"Hit@3 Precision:             {hit_at_3:.2f}%")
    print(f"Context Recall@{top_k}:         {recall_k:.2f}%")
    print(f"Mean Reciprocal Rank (MRR):  {mrr:.4f}")
    print("--------------------------------------------------")
    print(f"Avg Vector+BM25 Latency:     {avg_retrieval:.2f} ms")
    print(f"Avg Cross-Encoder Latency:   {avg_rerank:.2f} ms")
    print(f"P50 Retrieval Engine E2E:    {p50_total:.2f} ms")
    print(f"P95 Retrieval Engine E2E:    {p95_total:.2f} ms")
    print("==================================================\n")

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "tests", "eval_data.csv")
    run_extended_benchmark(csv_file)