from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Neo4j Database Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Model Configuration
    MODEL_NAME: str = "bert-base-uncased"
    SPACY_MODEL: str = "en_core_web_sm"
    
    # Knowledge Graph Configuration
    MAX_HOPS: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings() 