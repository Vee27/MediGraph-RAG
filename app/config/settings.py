from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "phi3:mini"
    ollama_embed_model: str = "nomic-embed-text"

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ChromaDB
    chroma_persist_dir: str = "./app/database/chroma"

    # LangSmith
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "mediagraph-rag"
    langchain_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()