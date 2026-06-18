import os
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from src.retrieval.vector_search import get_hybrid_retriever
from src.retrieval.graph_search import get_graph_chain
from src.config import Config


vector_retriever = get_hybrid_retriever()
graph_chain = get_graph_chain()
print("Retirvers loaded")


google_search = None
try:
    if Config.GOOGLE_API_KEY and Config.GOOGLE_CSE_ID:
        google_wrapper = GoogleSearchAPIWrapper(k=5)
        google_search = google_wrapper
    else:
        print("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID. Web search disabled.")
except Exception as e:
    print(f"Error initializing Google Search: {e}")

print("Backend is Ready.")

def vector_search_tool(query: str):
    """Searches internal documents (Vectors)."""
    try:
        print(f"[Vector Tool] Initiated: {query}")
        docs = vector_retriever.invoke(query)
        if not docs: return "No internal documents found."
        
        # Format with Source Metadata for Citation
        formatted = []
        for d in docs:
            source = d.metadata.get('source', 'Unknown')
            formatted.append(f"[Source: {source}]\n{d.page_content}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error in Vector Search: {e}"


def graph_search_tool(query: str):
    """Searches knowledge graph (Relationships)."""
    try:
        print(f"[Graph Tool] Initiated: {query}")
        response = graph_chain.invoke({"query": query})
        return str(response['result'])
    except Exception as e:
        return f"Error in Graph Search: {e}"



def web_search_tool(query: str):
    """Searches the live internet via Google using lightweight snippets (Fast)."""
    if not google_search:
        return "Web Search is disabled (Missing Keys)."
    
    try:
        print(f"[Web Tool] Fetching fast search snippets for: {query}")
        results = google_search.results(query, 4)
        
        formatted_results = []
        for i, res in enumerate(results):
            # Extract ONLY pre-compiled search card text metadata. Zero webpage scraping
            title = res.get('title', 'No Title')
            snippet = res.get('snippet', 'No Snippet available.')
            link = res.get('link', '')
            
            formatted_results.append(
                f"[Source {i+1}]\nTitle: {title}\nSnippet: {snippet}\nLink: {link}\n"
            )
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error in Web Search: {e}"