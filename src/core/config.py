from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings, loaded from the environment or .env file.
    Pydantic V2 handles validation and type coercion automatically.
    """
    gemini_api_key: str = ""
    app_name: str = "Research Agent"
    app_version: str = "1.0.0"
    max_results_cap: int = 30
    faiss_index_path: str = "./faiss_index"
    log_level: str = "INFO"

    # Pydantic V2 configuration for BaseSettings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance to be imported across the application
settings = Settings()