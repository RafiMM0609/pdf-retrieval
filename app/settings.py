from pydantic_settings import BaseSettings, SettingsConfigDict
class SettingsStruct(BaseSettings):
    openrouter_api_base: str
    api_key: str

    model_config = SettingsConfigDict(env_file=".env")
OPENROUTER_API_BASE = SettingsStruct.openrouter_api_base
API_KEY = SettingsStruct.api_key

# OPENROUTER_API_BASE = os.environ.get('OPENROUTER_API_BASE')
# API_KEY = os.environ.get('API_KEY')