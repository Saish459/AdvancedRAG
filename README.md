# Agent N - AI Powered Compliance Document Analysis

**Agent N** is an advanced Retrieval-Augmented Generation (RAG) system designed to analyze and retrieve information from compliance documents, specifically FAA Letters of Acceptance (LOA) and related regulatory materials. It combines vector search, knowledge graph retrieval, and web search capabilities to provide comprehensive document analysis through an interactive web interface.

## Key Features

- **Hybrid Search Architecture**: Combines vector similarity search (via Qdrant) and knowledge graph search (via NetworkX)
- **Smart Document Parsing**: Converts PDFs to markdown using LlamaParse for improved context understanding
- **Knowledge Graph Extraction**: Automatically extracts entities (organizations, documents, compliance plans, persons, dates) and their relationships
- **Multi-Modal Retrieval**: 
  - Internal vector search for semantic document matching
  - Graph search for relationship-based queries
  - Web search capability for live internet information
- **Web URL Analysis**: Detects and scrapes URLs from user queries for direct content analysis
- **Interactive Web UI**: Streamlit-based interface for easy document interaction
- **Network Visualization**: Interactive graph visualization of knowledge relationships

## Project Structure

```
.
├── app.py                          # Streamlit web application entry point
├── visualize_graph.py              # Network graph visualization utility
├── workspace_graph_index.gml       # Serialized knowledge graph
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration & environment variables
│   ├── pipeline.py                 # AdvancedRAGPipeline - core search logic
│   ├── utils.py                    # Shared utilities
│   │
│   ├── agent/
│   │   ├── graph.py                # LangGraph-based agent (currently commented)
│   │   └── tools.py                # Search tools (vector, graph, web)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py               # PDF parsing via LlamaParse
│   │   └── indexer.py              # Knowledge graph extraction & vector indexing
│   │
│   └── retrieval/
│       ├── vector_search.py        # Hybrid BM25 + semantic search
│       └── graph_search.py         # NetworkX knowledge graph queries
│
├── lib/                            # Frontend libraries
│   ├── bindings/utils.js
│   ├── tom-select/                 # Tom-select dropdown library
│   └── vis-9.1.2/                  # Vis.js network visualization
│
├── notebooks/
│   └── utility.ipynb               # Utility notebook
│
└── data/
    └── FAA.pdf
```

## Getting Started

### Prerequisites

- Python 3.10+
- API Keys:
  - NVIDIA API Key (for LLM & embeddings)
  - Llama Cloud API Key (for document parsing)
  - Google API Key & Search Engine ID (for web search)
- Databases:
  - Qdrant instance (for vector storage)

### Installation

1. **Clone the repository** 

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (create `.env` file):
   ```
   NVIDIA_API_KEY=your_nvidia_key
   LLAMA_CLOUD_API_KEY=your_llama_cloud_key
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_key
   GOOGLE_API_KEY=your_google_key
   GOOGLE_CSE_ID=your_cse_id
   ```

### Running the Application

#### 1. **Index Documents** (if adding new data)

Parse and index documents:
```bash
# From the root folder
python -m src.ingestion.indexer
# AND
python src/ingestion/parser.py
```

This will:
- Parse PDFs in `data/` folder to markdown (cached locally)
- Extract entities and relationships into NetworkX knowledge graph
- Create vector embeddings and store in Qdrant

#### 2. **Visualize the Knowledge Graph**

Generate and view an interactive network visualization:
```bash
python visualize_graph.py
```

This creates an interactive HTML visualization of all extracted entities and their relationships, with:
- Color-coded node types (Organization, Document, CompliancePlan, Person, Date, SourceFile)
- Node size based on entity importance
- Hoverable tooltips with entity details
- Dark theme optimized for readability

#### 3. **Launch the Web Interface**

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will:
- Start on `http://localhost:8501`
- Display a custom-styled dark theme UI
- Accept document queries and web URLs
- Show retrieved context and LLM responses with metrics

