import os
from pydantic_settings import BaseSettings  # modern replacement for pydantic BaseSettings


class Settings(BaseSettings):
    # === API Configuration ===
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Insulyn AI"  # Keep consistent with .env

    # === ML Model ===
    MODEL_PATH: str = "./data/best_model.pkl"

    # === Groq LLM Configuration ===
    GROQ_API_KEY: str  # required (from .env)
    LLM_MODEL_NAME: str  # required (from .env)
    LLM_TEMPERATURE: float = 0.6  # default, overridden by .env if present

    # === Application Environment ===
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # === Server Settings ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    class Config:
        # Ensure correct relative path: load from backend/.env
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


# ✅ Instantiate settings
settings = Settings()
