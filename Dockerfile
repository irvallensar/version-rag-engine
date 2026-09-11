FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache both lightweight ONNX models inside the image build
RUN python -c "from light_embed import TextEmbedding; \
    from fastembed.rerank.cross_encoder import TextCrossEncoder; \
    TextEmbedding('onnx-models/all-mpnet-base-v2-onnx'); \
    TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY src/ ./src/
COPY tests/ ./tests/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]