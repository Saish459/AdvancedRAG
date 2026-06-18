import os
import networkx as nx
from pyvis.network import Network

def run_beautified_visualization():
    print("Loading graph state...")
    if not os.path.exists("workspace_graph_index.gml"):
        print("❌ Error: workspace_graph_index.gml not found. Run your indexer first!")
        return
        
    G = nx.read_gml("workspace_graph_index.gml")
    
    net = Network(height="calc(100vh - 70px)", width="100%", bgcolor="#0d1117", font_color="#e6edf3", directed=True)
    
    color_map = {
        "Organization": {"background": "#ff5f1f", "border": "#ff4500"},
        "Document": {"background": "#00d26a", "border": "#00a854"},
        "CompliancePlan": {"background": "#38b6ff", "border": "#0096ff"},
        "Person": {"background": "#ffd700", "border": "#cca100"},
        "Date": {"background": "#ff66cc", "border": "#cc3399"},
        "SourceFile": {"background": "#8b5cf6", "border": "#6d28d9"}
    }
    
    print("Styling nodes and structural relationships...")
    for node, attrs in G.nodes(data=True):
        node_label = attrs.get("label", str(node))
        entity_type = attrs.get("type", "Unknown")
        
        styles = color_map.get(entity_type, {"background": "#6e7681", "border": "#484f58"})
        
        hover_tooltip = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 4px;">
            <strong style='color:{styles["background"]}; font-size: 14px;'>{node_label}</strong><br>
            <span style='color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;'>Type: {entity_type}</span>
        </div>
        """
        
        net.add_node(
            node, 
            label=node_label, 
            title=hover_tooltip,
            color=styles,
            size=28 if entity_type != "SourceFile" else 18,
            borderWidth=2,
            borderWidthSelected=4,
            font={"size": 12, "face": "Segoe UI, Helvetica, Arial", "color": "#f0f6fc"}
        )
        
    for source, target, edge_attrs in G.edges(data=True):
        rel_type = edge_attrs.get("type", "")
        
        if rel_type == "MENTIONED_IN":
            continue
            
        net.add_edge(
            source, 
            target, 
            label=rel_type, 
            arrows="to", 
            color={"color": "#30363d", "highlight": "#58a6ff", "hover": "#58a6ff"},
            font={"size": 10, "align": "top", "color": "#8b949e", "face": "Segoe UI", "strokeWidth": 0},
            width=2
        )
        
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "selectable": true
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.01,
          "springLength": 130,
          "springConstant": 0.07,
          "damping": 0.4
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 100
        }
      }
    }
    """)
    
    output_html = "graph_demo.html"
    net.save_graph(output_html)
    
    with open(output_html, "r") as file:
        html_content = file.read()

    custom_header_html = """
    <body style="margin: 0; padding: 0; overflow: hidden; background-color: #0d1117;">
        <header style="
            height: 60px; 
            background: linear-gradient(90deg, #161b22 0%, #0d1117 100%); 
            border-bottom: 1px solid #30363d; 
            display: flex; 
            align-items: center; 
            justify-content: space-between; 
            padding: 0 24px; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            position: relative;
            z-index: 999;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 10px; height: 10px; background-color: #58a6ff; border-radius: 50%; box-shadow: 0 0 10px #58a6ff;"></div>
                <h1 style="color: #f0f6fc; margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.5px;">Graph Knowledge Network</h1>
                <span style="background-color: #21262d; color: #8b949e; font-size: 11px; padding: 3px 8px; border-radius: 12px; border: 1px solid #30363d; margin-left: 8px;">AgentN GraphRAG Engine</span>
            </div>
            <div style="display: flex; gap: 16px; font-size: 12px; color: #8b949e;">
                <span style="display: flex; align-items: center; gap: 6px;"><span style="color: #ff5f1f;">●</span> Organization</span>
                <span style="display: flex; align-items: center; gap: 6px;"><span style="color: #00d26a;">●</span> Document</span>
                <span style="display: flex; align-items: center; gap: 6px;"><span style="color: #38b6ff;">●</span> Compliance Plan</span>
            </div>
        </header>
    """
    
    html_content = html_content.replace("<body>", custom_header_html)
    
    tooltip_style_override = """
    <style>
        div.vis-tooltip {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            color: #c9d1d9 !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
        }
        /* Custom styling for navigation buttons */
        div.vis-network div.vis-navigation div.vis-button {
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 4px !important;
            color: #f0f6fc !important;
        }
    </style>
    </style>
    """
    html_content = html_content.replace("</head>", f"{tooltip_style_override}</head>")

    with open(output_html, "w") as file:
        file.write(html_content)
        
    print(f"Dashboard UI generated")

if __name__ == "__main__":
    run_beautified_visualization()