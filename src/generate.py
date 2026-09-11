import re
import time
import os
from groq import Groq
from sentence_transformers import CrossEncoder
from embedder import Embedder
from db import DBConnection
from dotenv import load_dotenv

load_dotenv()

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# 1. INITIALIZE MODELS GLOBALLY (Loads into RAM only once at startup)
print("Loading Embedding and Reranking Models...")
global_embedder = Embedder()
global_db = DBConnection()
global_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
global_client = Groq()
print("Models loaded successfully.")

def infer_model_year(query_text: str) -> str | None:
    years = list(dict.fromkeys(YEAR_PATTERN.findall(query_text)))
    return years[0] if len(years) == 1 else None

def generate_answer(
    query_text: str,
    manufacturer: str = "BMW",
    model: str = "735i M Sport",
    model_year: str | None = None,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> tuple[str, str | None, list[dict], float, float]:
    
    if history is None:
        history = []

    # 2. REMOVE the local initialization here. Use the global variables instead.
    embedder = global_embedder
    db = global_db
    reranker = global_reranker
    client = global_client

    # Query Rewriter
    search_query = query_text
    if history:
        rewrite_prompt = (
            "Given the following conversation history and the user's latest question, "
            "rewrite the question into a standalone query. Do not answer it."
        )
        rewrite_messages = [{"role": "system", "content": rewrite_prompt}]
        for msg in history[-4:]:
            rewrite_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        rewrite_messages.append({"role": "user", "content": query_text})

        rewrite_resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=rewrite_messages,
            temperature=0.0,
        )
        search_query = rewrite_resp.choices[0].message.content.strip()

    effective_year = model_year or infer_model_year(search_query)

    # Retrieval & Reranking Benchmarking
    t_ret_start = time.perf_counter()
    query_vector = embedder.embed(search_query)
    initial_results = db.hybrid_search(
        query=search_query,
        query_embedding=query_vector,
        target_manufacturer=manufacturer,
        target_model=model,
        target_model_year=effective_year,
        top_k=20,
    )

    if not initial_results:
        t_ret_end = time.perf_counter() - t_ret_start
        return ("I cannot answer this based on the retrieved documentation.", effective_year, [], t_ret_end, 0.0)

    cross_inp = [[search_query, content] for _, content, _, _ in initial_results]
    cross_scores = reranker.predict(cross_inp)

    reranked_results = [
        res for _, res in sorted(
            zip(cross_scores, initial_results), key=lambda x: x[0], reverse=True
        )
    ][:top_k]
    t_ret_end = time.perf_counter() - t_ret_start

    context_blocks = []
    sources = []
    for i, (_, content, metadata, _) in enumerate(reranked_results, 1):
        src_title = f"{metadata.get('manufacturer', 'BMW')} {metadata.get('model', '')}"
        src_year = str(metadata.get('model_year', 'Unknown'))
        src_url = metadata.get('source_url', '')
        context_blocks.append(
            f"--- Document {i} [{src_title} ({src_year})] ---\n"
            f"Source URL: {src_url}\n{content}\n"
        )
        sources.append({
            "doc_id": i,
            "title": src_title,
            "year": src_year,
            "url": src_url,
            "snippet": content[:250].strip() + "..."
        })

    assembled_context = "\n".join(context_blocks)

    # Generation Benchmarking
    t_gen_start = time.perf_counter()
    system_prompt = (
        "You are an expert automotive documentation assistant. "
        "Answer the user's question using ONLY the provided BMW Indonesia documentation. "
        "Do not use outside knowledge. If not found, state exactly: "
        "'I cannot answer this based on the retrieved documentation.' "
        "Cite the document number (e.g. [Document 1]) when stating facts."
    )
    final_messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        final_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    final_messages.append({"role": "user", "content": f"Context:\n{assembled_context}\n\nQuestion: {query_text}"})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=final_messages,
        temperature=0.0,
    )
    t_gen_end = time.perf_counter() - t_gen_start

    return (response.choices[0].message.content, effective_year, sources, t_ret_end, t_gen_end)