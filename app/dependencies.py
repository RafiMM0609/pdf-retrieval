from functools import lru_cache
from qdrant_client import QdrantClient
from app.config import get_settings, Settings

@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_api_key:
        return QdrantClient(
            host=settings.qdrant_host, 
            port=settings.qdrant_port, 
            api_key=settings.qdrant_api_key
        )
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
