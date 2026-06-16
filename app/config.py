"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    """Runtime settings for Co-DM."""

    openrouter_api_key: str = Field("", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field("openai/gpt-4o-mini", alias="OPENROUTER_MODEL")
    app_name: str = Field("Co-DM", alias="APP_NAME")
    app_env: str = Field("dev", alias="APP_ENV")
    manuals_dir: str = Field("app/manuals", alias="MANUALS_DIR")
    chroma_dir: str = Field(".chroma/codm_manuals", alias="CHROMA_DIR")
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    retriever_k: int = Field(5, alias="RETRIEVER_K")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
