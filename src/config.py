import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    DATA_DIR = os.path.join(os.getcwd(), "data")
    EMBEDDING_MODEL = "nvidia/nv-embed-v1"
    LLM_MODEL = "meta/llama-4-maverick-17b-128e-instruct"

    @staticmethod
    def validate_keys():
        if not Config.NVIDIA_API_KEY:
            raise ValueError("Missing NVIDIA_API_KEY in .env")
        if not Config.LLAMA_CLOUD_API_KEY:
            raise ValueError("Missing LLAMA_CLOUD_API_KEY in .env")