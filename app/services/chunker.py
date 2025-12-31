from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] | None = None,
) -> List[str]:
    """
    General-purpose text chunker.
    Uses LangChain's RecursiveCharacterTextSplitter for robust splitting across separators.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)

def chunk_text_with_page_tracking(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Chunk text while preserving page number information.
    
    Args:
        pages_data: List of dicts with 'page_number' and 'text' keys.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.
        separators: List of separators for splitting.
    
    Returns:
        List of dicts with 'text', 'page_number', and 'page_numbers' (list) keys.
        'page_number' is the starting page, 'page_numbers' lists all pages the chunk spans.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks_with_pages = []
    
    for page_data in pages_data:
        page_num = page_data["page_number"]
        page_text = page_data["text"]
        
        if not page_text.strip():
            continue
        
        # Split text from this page
        page_chunks = splitter.split_text(page_text)
        
        for chunk in page_chunks:
            chunks_with_pages.append({
                "text": chunk,
                "page_number": page_num,
                "page_numbers": [page_num]  # Can be extended for multi-page chunks
            })
    
    return chunks_with_pages
