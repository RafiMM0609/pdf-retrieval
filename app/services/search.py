from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVector, SparseVectorParams, SparseIndexParams
import uuid
from collections import Counter
import math
from app.services.embedding import get_embedding_model

class BM25SparseEncoder:
    """Simple BM25-based sparse encoder for hybrid search."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def encode_documents(self, texts: List[str]) -> List[SparseVector]:
        """Encode texts into BM25 sparse vectors."""
        # Tokenize and compute term frequencies
        docs_tokens = [text.lower().split() for text in texts]
        docs_len = [len(tokens) for tokens in docs_tokens]
        avg_len = sum(docs_len) / len(docs_len) if docs_len else 1
        
        # Document frequency
        df = Counter()
        for tokens in docs_tokens:
            df.update(set(tokens))
        
        # Vocabulary
        vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        num_docs = len(texts)
        
        # Compute IDF
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        
        # Compute BM25 scores for each document
        sparse_vectors = []
        for tokens, doc_len in zip(docs_tokens, docs_len):
            tf = Counter(tokens)
            indices = []
            values = []
            for term, freq in tf.items():
                if term in vocab:
                    idx = vocab[term]
                    # BM25 formula
                    score = idf[term] * (freq * (self.k1 + 1)) / (
                        freq + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                    )
                    indices.append(idx)
                    values.append(score)
            sparse_vectors.append(SparseVector(indices=indices, values=values))
        
        return sparse_vectors

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
    """Create collection with hybrid search support (dense + sparse vectors)."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=distance),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams()
                ),
            },
        )

def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    embeddings: List[List[float]],
    sparse_vectors: List[SparseVector],
    payloads: List[Dict[str, Any]],
):
    """Upsert points with both dense and sparse vectors for hybrid search."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload=payload
        )
        for dense_vec, sparse_vec, payload in zip(embeddings, sparse_vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points)

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
    
    # Note: This sparse encoder is stateless and calculates IDF based on the query itself.
    # Ideally, it should use a pre-computed vocabulary/IDF from the corpus.
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
