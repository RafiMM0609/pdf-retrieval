from typing import List
from fastapi import HTTPException
import openai
import requests
import json
import settings

def generate_answer_with_openrouter(query: str, sources: List[dict], max_tokens: int = 500) -> str:
    """Generate answer using OpenRouter API."""
    # Validate max_tokens to prevent exceeding budget
    if max_tokens > 3000:
        max_tokens = 3000  # Cap at 3000 to stay under 3099 limit
    elif max_tokens < 50:
        max_tokens = 50  # Minimum reasonable tokens
    openai.api_base = settings.OPENROUTER_API_BASE
    openai.api_key = settings.API_KEY
    print("Open router key:", openai.api_key)
    
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not set")
    
    # Build context from sources
    context = "\n\n".join([
        f"Source: {s['filename']} (Page {s['page_number']})\nURL: {s['url']}\nText: {s['text']}"
        for s in sources
    ])
    
    prompt = f"""
    Based on the following context from PDF documents, answer the user's query.
    Include relevant download URLs in your answer where appropriate. Give response text format. 
    Do not pay attention to source paths that are local file paths.
    But pay attention to the page numbers and filenames.
    You may need page numbers and filenames to give detailed answers.
    Answer in short and concise way.
    Answer in text format.
    
    Context:
    {context}
    
    Query: {query}
    
    Answer:
    """
    
    try:
        response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
            # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": "openai/gpt-oss-20b",
            "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
            "max_tokens": max_tokens
        })
        )
        response_json = json.loads(response._content.decode('utf-8'))
        print("Response OpenRouter:\n", response_json)
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")