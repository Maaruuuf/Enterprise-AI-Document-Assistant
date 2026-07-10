"""
Centralized application configuration.

Uses pydantic-settings so values are validated at startup (fail fast) and can
be overridden via environment variables or a .env file without touching code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Paths ---
    DOCUMENTS_DIR: str = "documents"

    # --- Chunking ---
    CHUNK_SIZE_WORDS: int = 400
    CHUNK_OVERLAP_WORDS: int = 80

    # --- Embedding ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

    # --- Pinecone ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "enterprise-doc-assistant"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # --- Retrieval ---
    TOP_K_RETRIEVE: int = 15
    TOP_K_RERANK: int = 4
    CONFIDENCE_THRESHOLD: float = 0.5
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM ---
    GROQ_API_KEY: str
    GROQ_MODEL_PRIMARY: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_FALLBACKS: list[str] = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    ]
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 500

    # --- Session memory ---
    MAX_CONVERSATION_TURNS: int = 6          # how many past Q&A turns kept per session
    SESSION_TTL_SECONDS: int = 3600 * 6      # sessions expire after 6 hours of inactivity
    SESSION_NAME_MAX_WORDS: int = 6          # auto-generated session title length

    # --- API ---
    API_TITLE: str = "Enterprise AI Document Assistant"
    API_VERSION: str = "1.0.0"
    CORS_ALLOW_ORIGINS: list[str] = ["*"]  # tighten in production deployment


# Single shared settings instance — import this everywhere instead of
# re-instantiating Settings(), so env is parsed exactly once.
settings = Settings()