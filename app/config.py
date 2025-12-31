from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App Settings
    app_name: str = "PDF RAG API"
    debug: bool = False

    # OpenRouter Settings
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
