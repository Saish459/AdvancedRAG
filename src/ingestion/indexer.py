import os
import json
import concurrent.futures
from typing import List, Dict

# --- LangChain Core & Components ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_openai import ChatOpenAI  # Used for NVIDIA compatibility layer
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Database Clients & Math ---
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
from rank_bm25 import BM25Okapi

# --- Project Infrastructure ---
from src.config import Config
from src.utils import get_embedding_model
from src.ingestion.parser import load_data


# --- 1. LLM CONFIGURATION (NVIDIA via ChatOpenAI) ---
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


# --- 2. STRUCTURAL KNOWLEDGE EXTRACTION ---
def custom_graph_extraction(llm, chunk_text, source_file):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Aviation Compliance Analyst. Extract a Knowledge Graph from this FAA Letter of Acceptance (LOA).
        
        Extract the following entity types:
        - Organization (e.g., "Honeywell", "FAA", "Los Angeles ACO Branch", "Phoenix MIDO Section")
        - Document (e.g., "LOA0001LA", "AC 20-153B", "RTCA/DO-200A", "FMS Part Number Matrix")
        - CompliancePlan (e.g., "C72-1357-225", "NavDB Work Instructions", "QMS")
        - Person (e.g., "Erik Ringnes", "Mansour Rafat")
        - Date (e.g., "September 30, 2022", "January 3, 2005")

        Extract these relationship types:
        - COMPLIES_WITH (Organization -> Document)
        - DEFINED_IN (CompliancePlan -> Document)
        - REPORTS_TO (Organization -> Organization)
        - MANAGED_BY (CompliancePlan -> Person) or (Document -> Person)
        - EFFECTIVE_DATE (Document -> Date)
        - REVISION_OF (Document -> Document)

        Return strictly JSON:
        {{
            "nodes": [{{"id": "EntityName", "type": "EntityType"}}],
            "relationships": [{{"source": "EntityName", "target": "EntityName", "type": "RELATIONSHIP_TYPE"}}]
        }}

        Strict ID Normalization Rules:
        1. Standardize "Federal Aviation Administration" to "FAA".
        2. Standardize "Honeywell International Inc." or any variant to simply "Honeywell".
        3. Do NOT append revision text into the ID name string (e.g., "LOA0001LA Revision 3" must have the ID "LOA0001LA"). Keep base Document names identical across chunks so they merge seamlessly.
        4. Remove all whitespaces and special characters from IDs to maintain absolute string formatting consistency.
        
        Text to analyze:
        {text}
        """
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({"text": chunk_text})
        
        def clean_id(text):
            return text.strip()

        nodes = {}
        for n in result.get("nodes", []):
            cleaned_id = clean_id(n["id"])
            nodes[cleaned_id] = Node(id=cleaned_id, type=n["type"])
            
        rels = []
        for r in result.get("relationships", []):
            src_id = clean_id(r["source"])
            tgt_id = clean_id(r["target"])
            
            source_node = nodes.get(src_id, Node(id=src_id, type="Unknown"))
            target_node = nodes.get(tgt_id, Node(id=tgt_id, type="Unknown"))
            
            rels.append(Relationship(source=source_node, target=target_node, type=r["type"]))
        
        source_node = Node(id=source_file, type="SourceFile")
        all_nodes = list(nodes.values())
        all_nodes.append(source_node)
        
        for node in nodes.values():
            rels.append(Relationship(source=node, target=source_node, type="MENTIONED_IN"))

        doc = GraphDocument(
            nodes=all_nodes, 
            relationships=rels, 
            source=Document(page_content=chunk_text, metadata={"source": source_file})
        )
        return doc
        
    except Exception as e:
        print(f"Graph extraction failed for a chunk: {e}")
        return None


# --- 3. DOCUMENT CHUNKING ---
def split_documents(docs):
    print("✂️ Splitting documents (Raw Mode - 4000 chars)...")
    
    langchain_docs = []
    for doc in docs:
        langchain_docs.append(
            Document(
                page_content=doc['text'], 
                metadata={"source": doc['filename']}
            )
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(langchain_docs)
    print(f" Created {len(final_chunks)} chunks from {len(docs)} documents.")
    return final_chunks


# --- 4. LOCAL BM25 SPARSE VECTOR GENERATION ---
class LocalBM25Sparse:
    """Local BM25 sparse embedding that returns Qdrant SparseVector objects"""
    
    def __init__(self, **kwargs):
        self.bm25 = None
        
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """Convert documents to sparse vectors using BM25"""
        tokenized_docs = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        result = []
        for tokens in tokenized_docs:
            scores = self.bm25.get_scores(tokens)
            indices = [i for i, score in enumerate(scores) if score > 0]
            values = [float(scores[i]) for i in indices]
            result.append(SparseVector(indices=indices, values=values))
            
        return result
    
    def embed_query(self, text: str) -> SparseVector:
        """Convert query to sparse vector"""
        if not self.bm25:
            return SparseVector(indices=[], values=[])
            
        tokens = text.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        indices = [i for i, score in enumerate(scores) if score > 0]
        values = [float(scores[i]) for i in indices]
        
        return SparseVector(indices=indices, values=values)


# --- 5. HYBRID VECTOR INDEXING (QDRANT) ---
def index_vectors(chunks):
    print("Generating Hybrid Embeddings (Local BM25)...")
    dense_embeddings = get_embedding_model()
    sparse_embeddings = LocalBM25Sparse()
    
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY, timeout=60, check_compatibility=False)
    collection_name = "AgentN"
    
    try:
        test_embed = dense_embeddings.embed_query("test")
        vector_size = len(test_embed)
    except:
        vector_size = 4096

    if client.collection_exists(collection_name):
        print(f"🗑️ Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        sparse_vectors_config={"langchain-sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
    )

    qdrant_store = QdrantVectorStore(
        client=client, collection_name=collection_name, 
        embedding=dense_embeddings, sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID, sparse_vector_name="langchain-sparse"
    )

    print(f"Uploading {len(chunks)} chunks to Qdrant...")
    qdrant_store.add_documents(chunks, batch_size=4)
    print("Hybrid Vector Indexing Complete.")
    return qdrant_store


# --- 6. PROXY-SAFE IN-MEMORY GRAPH INDEXING (NETWORKX) ---
def index_graph(chunks):
    print("Extracting Knowledge Graph using NVIDIA Models (Memory Mode)...")
    
    llm = get_llm(temperature=0.0)
    
    # Instantiate the local in-memory LangChain NetworkX representation
    entity_graph = NetworkxEntityGraph()
    nx_g = entity_graph._graph  # Extract direct underlying NetworkX reference object
    
    graph_docs = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {
            executor.submit(custom_graph_extraction, llm, chunk.page_content, chunk.metadata['source']): chunk 
            for chunk in chunks
        }
        
        total = len(chunks)
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            completed += 1
            try:
                doc = future.result()
                if doc and (doc.nodes or doc.relationships):
                    graph_docs.append(doc)
                print(f"   Processed {completed}/{total} chunks")
            except Exception as exc:
                print(f"   Chunk generated an exception: {exc}")
    
    print(f"🔗 Extracted {len(graph_docs)} graph structures. Building NetworkX Instance...")
    
    # Map the custom extraction Document arrays into the local memory network
    for gdoc in graph_docs:
        for node in gdoc.nodes:
            nx_g.add_node(node.id, type=node.type)
            
        for rel in gdoc.relationships:
            nx_g.add_edge(
                rel.source.id, 
                rel.target.id, 
                type=rel.type
            )
            
    print(f"Graph Indexing Complete. NetworkX holds {entity_graph.get_number_of_nodes()} active layout nodes.")
    
    # Save network state to an offline GML asset file inside your project workspace folder
    entity_graph.write_to_gml("workspace_graph_index.gml")
    print("💾 Saved graph tracking state local to: workspace_graph_index.gml")
    
    return entity_graph


# --- 7. PIPELINE EXECUTION ENGINE ---
def run_indexing():
    raw_docs = load_data(Config.DATA_DIR)
    
    if not raw_docs: 
        print("No docs found.")
        return

    chunks = split_documents(raw_docs)
    index_vectors(chunks)
    index_graph(chunks) 


if __name__ == "__main__":
    run_indexing()