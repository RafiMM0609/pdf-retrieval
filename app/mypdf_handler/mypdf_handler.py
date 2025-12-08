from myvector_store.myvector_store import (
    upsert_chunks,
    ensure_collection
)
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from app.mychunker.mychunker import chunk_text_with_page_tracking
from app.myembeding.myembeding import get_embedding_model
from app.myvector_store.myvector_store import BM25SparseEncoder
from qdrant_client import QdrantClient 
from pypdf import PdfReader


def pdf_to_searchable_pdf(source_path: str) -> Tuple[bool, str]:
    """
    Convert a PDF to a searchable PDF using ocrmypdf.
    Falls back to copying original PDF if OCR fails.

    Returns: (status, dest_path_pdf)
    """
    src = Path(source_path)
    dest_path_pdf = str(src.with_stem(src.stem + "_converted"))
    stt = False
    try:
        subprocess.run([
            'ocrmypdf', 
            '--deskew', 
            '--clean', 
            '--force-ocr', 
            '--invalidate-digital-signatures',
            str(src), 
            dest_path_pdf
        ], check=True)
        stt = True
    except Exception as e:
        print(f"Convert file from pdf to text failed: {e}")
        shutil.copyfile(str(src), dest_path_pdf)
        stt = False
    return stt, dest_path_pdf


def extract_text(pdf_path: str) -> str:
    """Extract text from a (searchable) PDF using pypdf."""
    text_chunks = []
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_with_pages(pdf_path: str) -> List[Dict[str, any]]:
    """Extract text from a PDF with page number tracking.
    
    Returns:
        List of dicts with 'page_number' and 'text' keys.
    """
    pages_data = []
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages_data.append({
                "page_number": page_num,
                "text": page_text
            })
    return pages_data


def process_single_pdf(
    file_path: str,
    collection_name: str,
    model_name: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_api_key: Optional[str],
) -> dict:
    """Process a single PDF file and return processing details."""
    try:
        # OCR and extraction
        status, searchable_pdf = pdf_to_searchable_pdf(file_path)
        pages_data = extract_text_with_pages(searchable_pdf)
        
        if not pages_data or all(not p["text"].strip() for p in pages_data):
            return {
                "filename": Path(file_path).name,
                "chunks": 0,
                "status": "error",
                "message": "No text extracted from PDF"
            }
        
        # Chunking
        chunks_with_pages = chunk_text_with_page_tracking(
            pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        
        # Embeddings
        embedder = get_embedding_model(model_name)
        chunk_texts = [chunk["text"] for chunk in chunks_with_pages]
        vectors = embedder.embed(chunk_texts)
        vector_size = len(vectors[0]) if vectors else 0
        
        # Sparse vectors
        sparse_encoder = BM25SparseEncoder()
        sparse_vectors = sparse_encoder.encode_documents(chunk_texts)
        
        # Upsert to Qdrant
        if qdrant_api_key and qdrant_api_key.strip():
            client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)
        else:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
        ensure_collection(client, collection_name, vector_size)
        
        pdf_filename = Path(file_path).name
        payloads = [
            {
                "source_path": file_path,
                "filename": pdf_filename,
                "chunk_index": i,
                "page_number": chunk["page_number"],
                "page_numbers": chunk["page_numbers"],
                "text": chunk["text"],
            }
            for i, chunk in enumerate(chunks_with_pages)
        ]
        
        if vectors and sparse_vectors and payloads:
            upsert_chunks(client, collection_name, vectors, sparse_vectors, payloads)
            return {
                "filename": pdf_filename,
                "chunks": len(chunks_with_pages),
                "status": "success",
                "message": f"Processed {len(chunks_with_pages)} chunks"
            }
        else:
            return {
                "filename": pdf_filename,
                "chunks": 0,
                "status": "error",
                "message": "No vectors generated"
            }
    except Exception as e:
        return {
            "filename": Path(file_path).name,
            "chunks": 0,
            "status": "error",
            "message": str(e)
        }
