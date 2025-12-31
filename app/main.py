import os
import shutil
import tempfile
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient

from app.config import get_settings, Settings
from app.dependencies import get_qdrant_client
from app.schemas.api import QueryRequest, QueryResponse, UploadResponse
from app.services.llm import generate_answer_with_openrouter
from app.services.pdf import process_single_pdf
from app.services.search import perform_hybrid_search

app = FastAPI(
    title="PDF RAG API", 
    description="Retrieve and generate answers from PDF chunks using hybrid search and OpenRouter"
)

@app.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    client: QdrantClient = Depends(get_qdrant_client)
):
    """Endpoint to query documents and get AI-generated answer."""
    
    # Perform hybrid search
    sources = perform_hybrid_search(
        client=client,
        collection_name=request.collection_name,
        query_text=request.query,
        model_name=request.model_name,
        limit=request.limit or 5
    )
    
    if not sources:
        return QueryResponse(answer="No relevant information found.", sources=[], unique_files=[])
    
    # Generate answer with OpenRouter
    answer = generate_answer_with_openrouter(
        query=request.query, 
        sources=sources, 
        max_tokens=request.max_tokens or 500
    )
    
    # Extract unique filenames
    unique_files = list(set(source["filename"] for source in sources if source.get("filename")))
    
    return QueryResponse(answer=answer, sources=sources, unique_files=unique_files)

@app.post("/upload", response_model=UploadResponse)
async def upload_knowledge(
    files: List[UploadFile] = File(...),
    collection_name: str = "default_collection",
    model_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    client: QdrantClient = Depends(get_qdrant_client)
):
    """
    Upload multiple PDF files and process them into Qdrant collection.
    """
    # Validate files are PDFs
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith('.pdf')]
    
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files provided")
    
    files_details = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files temporarily
        saved_paths = []
        for file in pdf_files:
            if not file.filename:
                continue
                
            temp_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(temp_path)
            
        # Process the PDFs in parallel
        async def process_file(path):
            return await run_in_threadpool(
                process_single_pdf,
                client=client,
                file_path=path,
                collection_name=collection_name,
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if saved_paths:
            files_details = await asyncio.gather(*[process_file(path) for path in saved_paths])
    
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
            "POST /query": "Query documents and get AI-generated answers"
        }
    }
