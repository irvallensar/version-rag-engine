from light_embed import TextEmbedding

class Embedder:
    def __init__(self, model_name: str = "onnx-models/all-mpnet-base-v2-onnx"):
        print(f"Loading ONNX embedding model: {model_name}...")
        self.model = TextEmbedding(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode([text])[0]