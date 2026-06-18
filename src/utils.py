import os
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from src.config import Config

def get_llm(temperature=0.0):
    """
    Returns the NVIDIA LLM using the ChatOpenAI wrapper.
    This allows us to use 'Structured Output' which is required for Graph RAG.
    """
    return ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=Config.NVIDIA_API_KEY,
        model=Config.LLM_MODEL, 
        temperature=temperature,
        top_p=1.0,
    )

def get_embedding_model():
    """
    Returns the NVIDIA Embedding model using the native connector.
    """
    return NVIDIAEmbeddings(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=Config.NVIDIA_API_KEY,
        model=Config.EMBEDDING_MODEL,
        truncate="END"
    )