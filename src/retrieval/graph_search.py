import os
import warnings
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

# Clean up deprecation warnings in your VS Code terminal logs
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LocalGraphChain:
    """
    Mimics the exact interface contract of GraphCypherQAChain using a local 
    NetworkX index, ensuring total compatibility with tools.py without proxy blocks.
    """
    def __init__(self, gml_path: str = "workspace_graph_index.gml"):
        self.gml_path = gml_path
        self.entity_graph = None
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.gml_path):
            self.entity_graph = NetworkxEntityGraph.from_gml(self.gml_path)
        else:
            print(f"⚠️ Warning: {self.gml_path} not found. Run indexer pipeline first.")

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        The critical contract wrapper. 
        Accepts {"query": query} or {"question": query} and returns {"result": text}
        """
        # Handle variations in incoming tool keys gracefully
        query = inputs.get("query") or inputs.get("question") or ""
        
        if not self.entity_graph or not query:
            return {"result": "No graph database context available."}

        nx_g = self.entity_graph._graph
        matched_triplets = []
        query_lower = query.lower()
        
        # 1. Match node labels or identity IDs inside the query text string
        targeted_node_ids = []
        for node_id, attributes in nx_g.nodes(data=True):
            label = attributes.get("label", str(node_id)).lower()
            node_str = str(node_id).lower()

            if label in query_lower or node_str in query_lower:
                targeted_node_ids.append(node_id)
            elif ("loa" in query_lower or "acceptance" in query_lower) and (label == "loa0001la" or attributes.get("type") == "Document"):
                targeted_node_ids.append(node_id)

        # 2. Extract context triplets (Source -> Relation -> Target)
        for node_id in targeted_node_ids:
            source_label = nx_g.nodes[node_id].get("label", str(node_id))
            
            # Outbound paths
            for neighbor_id in nx_g.neighbors(node_id):
                edge_data = nx_g.get_edge_data(node_id, neighbor_id)
                edge_type = edge_data.get("type", "RELATED_TO")
                neighbor_label = nx_g.nodes[neighbor_id].get("label", str(neighbor_id))
                
                if edge_type == "MENTIONED_IN":
                    continue  # Filter out generic high-volume file paths
                    
                matched_triplets.append(f"[{source_label}] --({edge_type})--> [{neighbor_label}]")
                
            # Inbound paths for directed topologies
            if nx_g.is_directed():
                for predecessor_id in nx_g.predecessors(node_id):
                    edge_data = nx_g.get_edge_data(predecessor_id, node_id)
                    edge_type = edge_data.get("type", "RELATED_TO")
                    pred_label = nx_g.nodes[predecessor_id].get("label", str(predecessor_id))
                    
                    if edge_type == "MENTIONED_IN":
                        continue
                        
                    matched_triplets.append(f"[{pred_label}] --({edge_type})--> [{source_label}]")

        unique_triplets = list(set(matched_triplets))
        
        if not unique_triplets:
            result_context = "No direct graph relationships found matching query entities."
        else:
            result_context = "\n".join(unique_triplets)

        # Return structured map matching response['result'] contract exactly
        return {"result": result_context}


def get_graph_chain():
    """
    Factory method matching your tools.py import statement.
    Exposes an object that behaves like a LangChain runnable component.
    """
    chain_instance = LocalGraphChain()
    
    # Wrap it in a LangChain RunnableLambda to ensure type consistency across chains
    return RunnableLambda(chain_instance.invoke)


if __name__ == "__main__":
    # Internal test execution pass
    mock_chain = get_graph_chain()
    test_input = {"query": "What are the rules regarding Honeywell compliance plans?"}
    
    print("Testing local chain wrapper execution...")
    output = mock_chain.invoke(test_input)
    print("\nResult payload key evaluation:")
    print(output)