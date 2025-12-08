from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVector, SparseVectorParams, SparseIndexParams
import uuid
from collections import Counter
import math


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
    exists = False
    try:
        info = client.get_collection(collection_name)
        exists = True
    except Exception:
        exists = False
    if not exists:
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
