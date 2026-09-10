from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "claude-sonnet-5"
    retrieval_top_k: int = 4
    database_url: str

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
