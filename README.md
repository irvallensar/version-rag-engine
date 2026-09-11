# Production-Grade RAG System: Automotive Technical Intelligence

An optimized, full-stack Retrieval-Augmented Generation (RAG) architecture utilizing hybrid vector search and ONNX-based cross-encoder reranking to deliver highly accurate, document-grounded technical intelligence. 

This system was specifically engineered to query dynamic, version-dependent automotive specifications (e.g., BMW Indonesia documentation) while strictly adhering to a 512MB memory constraint in a serverless cloud environment.

## 🏗 System Architecture

The pipeline is split into two primary workflows:

1. **Ingestion Pipeline:** Scrapes dynamic web pages and PDFs (using Playwright and BeautifulSoup), parses them into contextual markdown chunks, generates 384-dimensional dense embeddings via `fastembed`, and stores them in a Supabase PostgreSQL database.
2. **Execution Pipeline:** Processes user queries through a contextual rewriting agent, retrieves relevant documents via a hybrid search algorithm, reranks the candidates using a Cross-Encoder, and streams a hallucination-free response via the Groq LLM API.

## 🛠 Tech Stack

* **Frontend:** Next.js, TailwindCSS, Vercel
* **Backend:** FastAPI, Python, Render
* **Vector Database:** PostgreSQL, `pgvector`, Supabase
* **Machine Learning:** `fastembed` (ONNX Runtime), Hugging Face (MiniLM-L6-v2)
* **LLM Engine:** Groq API (Llama / OSS Models)

## ⚡ Key Engineering Features

* **Hybrid Dense/Sparse Retrieval:** Merges semantic similarity (pgvector dense embeddings) with exact keyword matching (PostgreSQL Full-Text Search / BM25) using Reciprocal Rank Fusion (RRF) to handle highly specific alphanumeric queries (e.g., "M256 engine").
* **Cross-Encoder Reranking:** Implements an MS-MARCO Cross-Encoder to evaluate the logical relevance of retrieved chunks against the query before passing context to the LLM.
* **Extreme Memory Optimization:** Stripped all heavy PyTorch dependencies and replaced them with lightweight ONNX runtimes. This reduced the active RAM footprint from ~550MB to under 100MB, eliminating Out-Of-Memory (OOM) crashes on Render's free tier.
* **Automated CI/CD Evaluation:** Integrated GitHub Actions to run a 75-query benchmark suite against the retrieval pipeline on every push to ensure metric stability.

## 📊 Performance Benchmarks

Evaluated against a ground-truth dataset of 75 complex technical queries:

| Metric | Score | Note |
|--------|-------|------|
| **Context Recall@5** | 97.3% | Successfully retrieves the correct documentation chunk within the top 5 results. |
| **MRR (Mean Reciprocal Rank)** | 0.8778 | High precision in placing the most relevant document at the top of the stack. |
| **Retrieval Latency** | < 300ms | End-to-end database query and Cross-Encoder execution time. |

## 🚀 Getting Started (Local Development)

### Prerequisites
* Python 3.11+
* PostgreSQL database with `pgvector` enabled
* Groq API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/version-rag-engine.git](https://github.com/yourusername/version-rag-engine.git)
   cd version-rag-engine

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Environment Variables:**
Create a `.env` file in the root directory:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key

```


4. **Run the Database Ingestion (Optional):**
```bash
python src/pipeline.py

```


5. **Start the API Server:**
```bash
uvicorn src.api:app --reload --port 8000

```



## 📝 License

This project is licensed under the MIT License.

```

```
