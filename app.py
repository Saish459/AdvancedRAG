import streamlit as st
from src.pipeline import AdvancedRAGPipeline

st.set_page_config(
    page_title="Agent N",
    layout="wide",
)

if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<style>
    /* Global Font & Colors */
    .stApp {
        background-color: #0E1117;
    }
    
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF0000, #FF9900);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        padding-bottom: 20px;
    }
    
    .disclaimer-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Button Styling */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #f0f2f6;
        background-color: #1e232f;
    }
    
    /* Minimalist Metrics */
    .metrics-container {
        display: flex;
        gap: 15px;
        font-size: 0.8rem;
        color: #888;
        margin-top: 8px;
        font-style: italic;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


def show_welcome_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="gradient-text">Agent N</div>', unsafe_allow_html=True)
        st.markdown("### Your Advanced Research Assistant")
        st.markdown("---")

        st.markdown("""
        **Agent N** is an advanced RAG architecture designed for high-stakes regulatory and technical environments. 
        It combines State-of-the-Art retrieval strategies and advanced capabilities to provide grounded answers to users.
        """)
        
        st.markdown("""
        <div class="disclaimer-box">
            <strong>Disclaimer</strong><br><br>
            This system utilizes Large Language Models (LLMs) to synthesize information from provided documents and the web. 
            <br><br>
            1. <strong>Accuracy:</strong> While optimized for precision via citation, AI models can occasionally hallucinate or misinterpret complex data.<br>
            2. <strong>Verification:</strong> Critical compliance decisions should always be verified against the original source PDF linked in the citations.<br>
        </div>
        """, unsafe_allow_html=True)
        
        ack = st.checkbox("I acknowledge the limitations of this AI-driven analysis.")
        
        # Entry Button
        if st.button("Initialize System", type="primary", disabled=not ack):
            st.session_state.terms_accepted = True
            st.rerun()

def show_main_interface():
    with st.sidebar:
        st.image("notebooks/logo.svg", width=280)  
        st.markdown("---")      
        st.caption("v1.0 • Local Environment")
        
        with st.expander("Tech Stack", expanded=False):
            st.markdown("""
            - **LangChain:** Orchestration
            - **Qdrant:** Vector Database
            - **NetworkX Graph:** In-Memory Graph
            - **Streamlit:** UI/UX
            - **Google Search:** Live Web Results
            """)
            with st.expander("**Technical Aspects**", expanded=False):
                st.markdown("""
                - **Dense Vectors:** Semantic Understanding (Nvidia's nv-embed).
                - **Sparse Vectors:** Keyword Precision (Local BM25 Alignment).
                - **Knowledge Graph:** Relationship Mapping (NetworkX In-Memory).
                - **Intent Classification:** Routes queries to Docs, Web, or Chat.
                - **Query Rewriting:** Transforms vague follow-ups into precise search terms.
                - **Context Awareness:** Remembers previous turns.
                - **Source Tracking:** Every claim is linked to a PDF filename.
                - **Near-Zero Hallucination:** Strict constraints on guessing.
                - **Google Search API:** For real-time web data.
                """)
        
        with st.expander("Chat History", expanded=False):
            if not st.session_state.messages:
                st.caption("No conversation history yet.")
            else:
                for msg in st.session_state.messages:
                    role_lbl = "User" if msg["role"] == "user" else "Agent"
                    preview = (msg["content"][:40] + "...") if len(msg["content"]) > 40 else msg["content"]
                    st.markdown(f"**{role_lbl}:** {preview}")
                    st.markdown("---")

        if st.button("Clear Messages", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="gradient-text" style="font-size: 2.5rem;">Agent N</div>', unsafe_allow_html=True)
    st.caption("Your Advanced Research Partner")

    @st.cache_resource
    def get_pipeline():
        return AdvancedRAGPipeline()
    
    pipeline = get_pipeline()

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            metrics = message.get("metrics")
            if metrics:
                st.markdown(
                    f"""
                    <div class="metrics-container">
                        <span>{metrics['time']}s</span>
                        <span>•</span>
                        <span>{metrics['tokens']} tokens</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            if message.get("context_preview"):
                with st.expander("View Source Context"):
                    st.code(message["context_preview"], language="markdown")

    if prompt := st.chat_input("Ask Agent N anything..."):
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🤖"):
            with st.status("Processing query...", expanded=True) as status:
                history = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"][-5:]
                
                result = pipeline.run(prompt, history)
                
                if result['category'] == "DIRECT_URL":
                    st.write(f"**Detected URL:** Link found in query.")
                    st.write("**Strategy:** `DIRECT_SCRAPE`")
                    st.write("Scraping and analyzing page content...")
                else:
                    st.write(f"**Refined:** `{result['rewritten_query']}`")
                    if "COMPLIANCE" in result['category'] or "INTERNAL" in result['category']:
                        st.write("**Source:** Internal Knowledge Base (Dense + BM25 + NetworkX)")
                    elif "WEB" in result['category']:
                        st.write("**Source:** Live Web Search")
                    else:
                        st.write("**Mode:** Conversational")
                
                status.update(label="Complete", state="complete", expanded=False)

            st.markdown(result["answer"])
            
            metrics = result.get("metrics")
            if metrics:
                st.markdown(
                    f"""
                    <div class="metrics-container">
                        <span>{metrics['time']}s</span>
                        <span>•</span>
                        <span>{metrics['tokens']} tokens</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "context_preview": result.get("context_preview"),
                "metrics": metrics
            })

if st.session_state.terms_accepted:
    show_main_interface()
else:
    show_welcome_screen()