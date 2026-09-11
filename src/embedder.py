from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()