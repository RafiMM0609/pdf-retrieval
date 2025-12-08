from typing import List, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    collection_name: str
    limit: Optional[int] = 5
    model_name: Optional[str] = None
    max_tokens: Optional[int] = 500

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]  # List of {"url": str, "filename": str, "page_number": int, "text": str, "score": float}
    unique_files: List[str]  # List of unique filenames involved in the sources

class UploadResponse(BaseModel):
    message: str
    collection_name: str
    files_processed: int
    files_details: List[dict]  # List of {"filename": str, "chunks": int, "status": str}