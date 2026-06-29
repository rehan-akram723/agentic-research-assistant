from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment / .env file."""

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4.1-nano"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    app_name: str = "Research Agent"
    app_version: str = "1.0.0"

    max_results_cap: int = 30
    faiss_index_path: str = "./faiss_index"
    log_level: str = "INFO"

    # Hugging Face token for authenticated model downloads (gated models, higher rate limits).
    # Also honoured by sentence-transformers automatically when set as HF_TOKEN env var.
    hf_token: str = ""

    # Pydantic V2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()