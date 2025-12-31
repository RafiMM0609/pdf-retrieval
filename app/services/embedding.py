from typing import List
from sentence_transformers import SentenceTransformer
from functools import lru_cache

class EmbeddingModel:
    """Simple strategy interface for embeddings."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self.model.encode(texts, convert_to_numpy=True)]

@lru_cache
def get_embedding_model(name: str | None = None) -> EmbeddingModel:
    """
    Factory to get an embedding model. For now supports sentence-transformers.
    """
    return SentenceTransformerEmbeddings(model_name=name or "BAAI/bge-m3")
