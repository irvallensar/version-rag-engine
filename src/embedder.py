from fastembed import TextEmbedding

class Embedder:
    def __init__(self):
        self.model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=1)

    def embed(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()