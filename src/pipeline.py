import time
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.utils import get_llm
from src.agent.tools import vector_search_tool, graph_search_tool, web_search_tool


class AdvancedRAGPipeline:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        
    def run(self, user_query: str, chat_history: list):
        start_time = time.time()
        print(f"\nPipeline Processing: '{user_query}'")

        # --- 1. DIRECT URL DETECTION & SCRAPE ---
        url_match = re.search(r'(https?://\S+)', user_query)
        
        if url_match:
            target_url = url_match.group(0)
            print(f"   URL Detected: {target_url}")
            
            try:
                loader = WebBaseLoader(target_url)
                docs = loader.load()
                scraped_content = docs[0].page_content[:10000]
                print("   Scraped successfully.")
                
                system_instruction = f"""
                You are 'Agent N', an expert web analyst.
                The user has provided a specific URL to analyze: {target_url}
                
                RULES:
                1. Answer the user's question using ONLY the content scraped from the URL below.
                2. If the answer is not in the page content, say so.
                3. Be concise and professional.
                """
                
                final_prompt = ChatPromptTemplate.from_template(
                    """
                    {system_instruction}
                    
                    SCRAPED CONTENT:
                    {context}
                    
                    USER QUESTION: {question}
                    """
                )
                
                chain = final_prompt | self.llm
                response_msg = chain.invoke({
                    "system_instruction": system_instruction,
                    "context": scraped_content,
                    "question": user_query
                })
                
                answer_text = response_msg.content
                token_usage = response_msg.response_metadata.get('token_usage', {})
                total_tokens = token_usage.get('total_tokens', 0)

                end_time = time.time()
                return {
                    "answer": answer_text,
                    "category": "DIRECT_URL",
                    "rewritten_query": user_query,
                    "context_preview": f"Source: {target_url}\n\n{scraped_content[:500]}...",
                    "metrics": {
                        "time": round(end_time - start_time, 2), 
                        "tokens": total_tokens
                    }
                }
                
            except Exception as e:
                print(f"   Scraping failed: {e}")
                pass

        # --- 2. QUERY REWRITING WITH TRUE CONTEXT AWARENESS ---
        rewritten_query = user_query
        if chat_history:
            print("   Rewriting query using chat history context...")
            rewrite_prompt = ChatPromptTemplate.from_template(
                """
                Given the following conversation history and a follow-up question, 
                rewrite the follow-up question to be a standalone search query.
                
                CRITICAL RULES:
                1. Do NOT guess or expand technical acronyms randomly. Keep terms like "LOA" as "LOA" or use "Letter of Acceptance". Do NOT use "Letter of Agreement".
                2. Keep the rewritten output strictly to 1 line.

                Chat History:
                {chat_history}
                
                Follow-up Question: {question}
                
                Standalone Query:
                """
            )
            rewriter = rewrite_prompt | self.llm | StrOutputParser()
            # Pass the historical list to make pronouns resolvable
            rewritten_query = rewriter.invoke({
                "chat_history": "\n".join(chat_history),
                "question": user_query
            }).strip()
            print(f"   Rewritten Standalone Query: '{rewritten_query}'")

        # --- 3. INTENT CLASSIFICATION ROUTER ---
        print("   Routing...")
        router_prompt = ChatPromptTemplate.from_template(
            """
            Classify the query into ONE category:
            1. COMPLIANCE: Questions about LOAs, Regulations, Part Numbers, specific documents, internal Honeywell files.
            2. WEB: Questions about Live events, Competitors, Stock prices, Well known People, General Knowledge, Company Spin Off, NOT in the docs.
            3. CHAT: Greetings, compliments, meta-questions.
            
            Return ONLY the category name.
            
            Query: {query}
            """
        )
        router = router_prompt | self.llm | StrOutputParser()
        category = router.invoke({"query": rewritten_query}).strip().upper()
        print(f"   Intent Routed To: {category}")

        # --- 4. DATA RETRIEVAL EXECUTION MATRIX ---
        context = ""
        
        if "COMPLIANCE" in category:
            print("   Searching Internal Vector + Graph Indexes...")
            vec_res = vector_search_tool(rewritten_query)
            
            # FIXED: Aligned parameter argument to pass variable tracking rewritten_query safely
            graph_res = graph_search_tool(rewritten_query) 
            
            context = f"INTERNAL DOCUMENTS (High Trust):\n{vec_res}\n\nKNOWLEDGE GRAPH (Relationships):\n{graph_res}"
            
        elif "WEB" in category:
            print("   Searching Google Engine...")
            web_res = web_search_tool(rewritten_query)
            context = f"WEB SEARCH RESULTS (External):\n{web_res}"
            
        else:
            print("   Chat Mode Active")
            context = "No context needed. User is chatting."

        # --- 5. DYNAMIC GENERATION PROMPT MATRICES ---
        print("   Generating Answer...")
        
        if "COMPLIANCE" in category:
            system_instruction = """
            You are 'Agent N', an expert Aviation Compliance Assistant.
            Your context includes raw document text snippets (High Trust) and high-level relationships (Knowledge Graph).
            
            STRICT RULES:
            1. **PRIORITIZE RAW TEXT:** The 'INTERNAL DOCUMENTS' section contains the exact literal text of the FAA letter. Use it as your primary truth for explicit text questions (like addressees, names, dates, and sentences).
            2. **USE GRAPH FOR STRUCTURE:** Use the 'KNOWLEDGE GRAPH' to understand how entities connect, report, or comply with regulations.
            3. **CITATIONS:** You must cite your sources! 
               - If the fact comes from the raw text, use `[Source: filename]`.
               - If the fact comes from graph triplets, use `[Source: KNOWLEDGE GRAPH]`.
            4. **ACCURACY:** Do not guess. If the information is genuinely missing from both the text and the graph, say "I cannot find that information in the internal documents."
            
            TONE: Professional, analytical, auditor-like.
            """
        
        elif "WEB" in category:
            system_instruction = """
            You are 'Agent N', a sophisticated AI researcher with real-time web access.
            
            GOAL: Synthesize the provided search results into a high-quality, structured answer.
            
            STRICT RULES:
            1. **CITATIONS:** You MUST cite your sources at the end of the sentence using the format `[Source X]`.
               - Example: "Honeywell announced a new CEO in 2024 [Source 1]."
               - Provide the matching title/link list at the very bottom of your response without fail.
            2. **STRUCTURE:** Use **Bold Headers** and Bullet points to break down complex topics.
            3. **NEUTRALITY:** Report facts as found in the search results. Do not invent information.
            
            TONE: Professional, objective, and journalistic.
            """
        
        else:
            system_instruction = """
            You are 'Agent N', a friendly AI assistant.
            Engage in normal conversation. Be polite and concise.
            """

        final_prompt = ChatPromptTemplate.from_template(
            """
            {system_instruction}
            
            Context:
            {context}
            
            User Question: {question}
            """
        )

        chain = final_prompt | self.llm
        response_msg = chain.invoke({
            "system_instruction": system_instruction,
            "context": context, 
            "question": rewritten_query
        })
        
        answer_text = response_msg.content
        
        token_usage = response_msg.response_metadata.get('token_usage', {})
        total_tokens = token_usage.get('total_tokens', 0)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "answer": answer_text,
            "category": category,
            "rewritten_query": rewritten_query,
            "context_preview": context[:200] + "...",
            "metrics": {
                "time": round(execution_time, 2),
                "tokens": total_tokens
            }
        }