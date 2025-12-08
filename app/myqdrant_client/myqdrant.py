from typing import List, Optional
from qdrant_client import QdrantClient
from app.myembeding.myembeding import get_embedding_model
from app.myvector_store.myvector_store import BM25SparseEncoder

def perform_hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    model_name: Optional[str],
    limit: int
) -> List[dict]:
    """Perform hybrid search and return results with URLs."""
    embedder = get_embedding_model(model_name)
    dense_vector = embedder.embed([query_text])[0]
    
    sparse_encoder = BM25SparseEncoder()
    sparse_vector = sparse_encoder.encode_documents([query_text])[0]
    
    results = client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        limit=limit,
    )
    
    sources = []
    for point in results.points:
        payload = point.payload
        source_path = payload.get("source_path", "")
        filename = payload.get("filename", "")
        page_number = payload.get("page_number", "N/A")
        # Assume source_path is a local file; convert to download URL if needed
        # For now, use file:// URL or placeholder
        url = f"file://{source_path}" if source_path else "N/A"
        sources.append({
            "url": url,
            "filename": filename,
            "page_number": page_number,
            "source_path": source_path,
            "text": payload.get("text", "")[:300],  # Truncate for brevity
            "score": float(point.score)
        })
    return sources