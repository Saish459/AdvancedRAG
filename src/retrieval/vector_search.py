import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from src.config import Config
from src.utils import get_embedding_model
from src.ingestion.indexer import LocalBM25Sparse
from src.ingestion.parser import load_data

def get_hybrid_retriever():
    dense_embeddings = get_embedding_model()
    
    # Initialize the local sparse model
    sparse_embeddings = LocalBM25Sparse()
    
    # Read the raw files quickly so the BM25Okapi token matrix matches your database
    try:
        raw_docs = load_data(Config.DATA_DIR)
        if raw_docs:
            texts = [doc['text'] for doc in raw_docs]
            # Warm up the token weights matrix in memory!
            sparse_embeddings.embed_documents(texts)
            print("Local BM25 Sparse vocabulary matrix loaded successfully.")
    except Exception as e:
        print(f"Could not warm up local BM25 vocabulary weights: {e}")

    client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="AgentN",
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="langchain-sparse"
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 3})