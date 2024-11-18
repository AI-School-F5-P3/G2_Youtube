from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./youtube_messages.db"
    MODEL_NAME: str = "distilbert-base-uncased"

    class Config:
        env_file = ".env"


settings = Settings()

