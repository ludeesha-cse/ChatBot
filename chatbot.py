import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_community.utilities import GoogleSerperAPIWrapper

# Local RAG retriever
from retrieve import retrieve_context


# Load env
load_dotenv()


class ChatbotState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    search_results: Optional[str]
    needs_search: bool
    rag_context: Optional[str]


# Initialize LLM (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Add it to your .env file.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)


# Initialize Serper
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
search = None
if SERPER_API_KEY:
    try:
        search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    except Exception:
        # Fallback to no search if initialization fails
        search = None


def should_search(state: ChatbotState) -> str:
    """Only search the web if RAG context is insufficient for the query."""
    if not search:
        return "chatbot_node"

    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # First, try to get RAG context
    try:
        rag_context = retrieve_context(user_message, k=3)
        
        # Quick check: if context is clearly insufficient, go directly to search
        if (not rag_context or 
            len(rag_context.strip()) < 100 or 
            "No relevant documents found" in rag_context or 
            "Retrieval error" in rag_context):
            
            state["rag_context"] = ""  # Clear irrelevant context
            return "search_node"
        
        # We have substantial context, now check if it's relevant
        # Use LLM to quickly check if the context is actually relevant to the question
        relevance_check = f"""
        Question: {user_message}
        Context: {rag_context[:500]}...
        
        Is this context relevant to answer the question? Reply with only 'YES' or 'NO'.
        """
        
        try:
            relevance_resp = llm.invoke([HumanMessage(content=relevance_check)])
            is_relevant = (relevance_resp.content or "").strip().upper().startswith("YES")
            
            if is_relevant:
                state["rag_context"] = rag_context
                return "chatbot_node"
            else:
                state["rag_context"] = ""  # Clear irrelevant context
                return "search_node"
        except Exception as e:
            state["rag_context"] = rag_context
            return "chatbot_node"
            
    except Exception as e:
        state["rag_context"] = ""
        return "search_node"


def search_node(state: ChatbotState) -> dict:
    """Run Serper search and format top results."""
    if not search:
        return {"search_results": None, "needs_search": False}

    query = state["messages"][-1].content if state["messages"] else ""
    try:
        results = search.results(query)
        formatted = ""
        if isinstance(results, dict) and "organic" in results:
            for item in results["organic"][:3]:
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "")
                formatted += f"- {title}: {snippet} [Source: {link}]\n"
        if not formatted:
            formatted = "No relevant search results found."
        return {"search_results": formatted, "needs_search": False}
    except Exception as e:
        return {"search_results": f"Error fetching search results: {e}", "needs_search": False}


def chatbot_node(state: ChatbotState) -> dict:
    """Generate the assistant reply using chat history, RAG context and optional search results."""
    user_message = state["messages"][-1].content if state["messages"] else ""

    # Use RAG context from should_search if available, otherwise retrieve it
    rag_context = state.get("rag_context", "")
    if not rag_context:
        try:
            rag_context = retrieve_context(user_message, k=3)
        except Exception as e:
            rag_context = ""

    search_results = state.get("search_results") or ""

    # Determine if we should use RAG context or search results
    has_relevant_rag = rag_context and len(rag_context.strip()) > 50
    has_search_results = search_results and "No relevant search results found" not in search_results

    if has_relevant_rag and not has_search_results:
        # RAG-only response
        system_instructions = (
            "You are a helpful assistant. Answer the question using the provided document context. "
            "Be concise and accurate."
        )
        composed_prompt = f"Question: {user_message}\n\nContext: {rag_context}\n\nAnswer based on the context above."
        
    elif has_search_results and not has_relevant_rag:
        # Search-only response
        system_instructions = (
            "You are a helpful assistant. Answer the question directly using only the provided search results. "
            "Do not mention document context, primary source, secondary source, or explain your sources of information. "
            "Just provide a direct answer and cite sources inline when appropriate."
        )
        composed_prompt = f"{user_message}\n\nSearch Results:\n{search_results}\n\nProvide a direct answer to the question using the search results."
        
    elif has_relevant_rag and has_search_results:
        # Both RAG and search available
        system_instructions = (
            "You are a helpful assistant. Prioritize the document context, but supplement with search results if needed. "
            "Cite sources when using search results."
        )
        composed_prompt = (
            f"Question: {user_message}\n\n"
            f"Primary Source - Document Context:\n{rag_context}\n\n"
            f"Secondary Source - Search Results:\n{search_results}\n\n"
            f"Answer using the document context first. Only use search results to supplement if needed."
        )
    else:
        # Neither available - fallback
        system_instructions = "You are a helpful assistant. Answer the question based on your general knowledge."
        composed_prompt = f"Question: {user_message}"

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=composed_prompt),
    ]

    try:
        response = llm.invoke(messages)
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "needs_search": False,
        }
    except Exception as e:
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error generating response: {e}")],
            "needs_search": False,
        }


def start_node(state: ChatbotState) -> dict:
    return {"needs_search": True}


# Build LangGraph
workflow = StateGraph(ChatbotState)
workflow.add_node("start_node", start_node)
workflow.add_node("search_node", search_node)
workflow.add_node("chatbot_node", chatbot_node)

workflow.set_entry_point("start_node")
workflow.add_conditional_edges(
    "start_node", should_search, {"search_node": "search_node", "chatbot_node": "chatbot_node"}
)
workflow.add_edge("search_node", "chatbot_node")
workflow.add_edge("chatbot_node", END)

app = workflow.compile()


def run_chatbot():
    """Simple CLI loop for chatting."""
    state: ChatbotState = {"messages": [], "search_results": None, "needs_search": True, "rag_context": None}
    print("Chatbot started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        state["messages"].append(HumanMessage(content=user_input))
        state["needs_search"] = True
        state["rag_context"] = None  # Reset for new query
        state = app.invoke(state)
        print("Bot:", state["messages"][-1].content)


if __name__ == "__main__":
    run_chatbot()
