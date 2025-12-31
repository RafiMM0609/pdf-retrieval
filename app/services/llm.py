from typing import List
from fastapi import HTTPException
from openai import OpenAI
from app.config import get_settings

settings = get_settings()

def generate_answer_with_openrouter(query: str, sources: List[dict], max_tokens: int = 500) -> str:
    """Generate answer using OpenRouter API."""
    
    # Validate max_tokens
    max_tokens = max(50, min(max_tokens, 3000))
    
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not set")
    
    client = OpenAI(
        base_url=settings.openrouter_api_base,
        api_key=settings.openrouter_api_key,
    )
    
    # Build context from sources
    context_parts = []
    for s in sources:
        source_info = f"Source: {s.get('filename', 'Unknown')} (Page {s.get('page_number', 'N/A')})"
        url_info = f"URL: {s.get('url', 'N/A')}"
        text_info = f"Text: {s.get('text', '')}"
        context_parts.append(f"{source_info}\n{url_info}\n{text_info}")
    
    context = "\n\n".join(context_parts)
    
    system_prompt = """
    You are a helpful assistant. Based on the provided context from PDF documents, answer the user's query.
    Include relevant download URLs in your answer where appropriate.
    Do not pay attention to source paths that are local file paths.
    Pay attention to the page numbers and filenames.
    Answer in a short and concise way.
    """

    user_prompt = f"""
    Context:
    {context}
    
    Query: {query}
    
    Answer:
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", # Consider making this configurable
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
