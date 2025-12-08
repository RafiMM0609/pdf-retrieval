from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List, Optional
import os
import tempfile
import shutil
from pathlib import Path
from qdrant_client import QdrantClient
from app.mypdf_handler.mypdf_handler import process_single_pdf
from app.schemas.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse
)
from app.myopenrouter.myopenrouter import generate_answer_with_openrouter
from app.myqdrant_client.myqdrant import perform_hybrid_search

app = FastAPI(title="PDF RAG API", description="Retrieve and generate answers from PDF chunks using hybrid search and OpenRouter")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Endpoint to query documents and get AI-generated answer.
    
    Parameters:
    - query: The search query string
    - collection_name: Name of the Qdrant collection to search
    - limit: Maximum number of search results to retrieve (default: 5)
    - model_name: Embedding model name (optional)
    - max_tokens: Maximum number of tokens in AI response (default: 500, max: 3000)
    """
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # Initialize QdrantClient - only pass api_key if it's not empty
    if qdrant_api_key and qdrant_api_key.strip():
        client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)
    else:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    # Perform hybrid search
    sources = perform_hybrid_search(
        client=client,
        collection_name=request.collection_name,
        query_text=request.query,
        model_name=request.model_name,
        limit=request.limit
    )
    
    if not sources:
        return QueryResponse(answer="No relevant information found.", sources=[], unique_files=[])
    
    # Generate answer with OpenRouter
    answer = generate_answer_with_openrouter(request.query, sources, request.max_tokens)
    
    # Extract unique filenames
    unique_files = list(set(source["filename"] for source in sources if source["filename"]))
    
    return QueryResponse(answer=answer, sources=sources, unique_files=unique_files)

@app.post("/upload", response_model=UploadResponse)
async def upload_knowledge(
    files: List[UploadFile] = File(...),
    collection_name: str = "default_collection",
    model_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Upload multiple PDF files and process them into Qdrant collection.
    
    Parameters:
    - files: List of PDF files to upload
    - collection_name: Name of the Qdrant collection to store embeddings
    - model_name: Embedding model name (default: BAAI/bge-m3)
    - chunk_size: Size of text chunks (default: 1000)
    - chunk_overlap: Overlap between chunks (default: 200)
    """
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # Validate files are PDFs
    pdf_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a PDF. Only PDF files are accepted."
            )
        pdf_files.append(file)
    
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files provided")
    
    files_details = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files temporarily and process them
        for file in pdf_files:
            temp_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the PDF
            result = process_single_pdf(
                file_path=temp_path,
                collection_name=collection_name,
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                qdrant_api_key=qdrant_api_key,
            )
            files_details.append(result)
    
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
    
    successful_files = sum(1 for f in files_details if f["status"] == "success")
    
    return UploadResponse(
        message=f"Processed {successful_files}/{len(pdf_files)} files successfully",
        collection_name=collection_name,
        files_processed=successful_files,
        files_details=files_details
    )

@app.get("/")
async def root():
    return {
        "message": "PDF RAG API is running.",
        "endpoints": {
            "POST /upload": "Upload multiple PDF files to knowledge base",
            "POST /query": "Query documents and get AI-generated answers (supports limit and max_tokens parameters)"
        }
    }
