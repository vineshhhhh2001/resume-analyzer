from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str = "sqlite+aiosqlite:///./storage/resume_analyzer.db"
    upload_max_size_mb: int = 10
    cors_origins: str = "http://localhost:5173"
    host: str = "0.0.0.0"
    port: int = 8000
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ai_provider: str = "local" # "openai", "gemini", "groq", or "local"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
