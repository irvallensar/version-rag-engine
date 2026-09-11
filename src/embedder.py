from fastembed import TextEmbedding

class Embedder:
    def __init__(self):
        # Uses the exact same 384-dimensional MiniLM model in ONNX format (~90MB on disk, ~30MB RAM)
        self.model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()