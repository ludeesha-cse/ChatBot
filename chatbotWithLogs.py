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
    rag_context: Optional[str]
    requires_search: bool


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
    """Always go to chatbot first - retrieve RAG context and let LLM decide."""
    print("Starting workflow - retrieving RAG context...")
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # Retrieve RAG context
    try:
        rag_context = retrieve_context(user_message, k=3)
        state["rag_context"] = rag_context if rag_context else ""
        print(f"Retrieved context length: {len(state['rag_context'])} characters")
        if state["rag_context"]:
            print(f" Context preview: {state['rag_context'][:100]}...")
        else:
            print(" No relevant context found")
    except Exception as e:
        print(f" Error retrieving context: {e}")
        state["rag_context"] = ""
    
    state["requires_search"] = False
    print(" Going to chatbot_node")
    return "chatbot_node"


def check_if_search_needed(state: ChatbotState) -> str:
    """Check if the LLM's response indicates it needs search."""
    print("Checking if search is needed...")
    
    if not search:
        print("No search API available, ending")
        return END
        
    if not state.get("requires_search", False):
        print(" No search required, ending conversation")
        return END
    
    # Get the last AI response
    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message or not isinstance(last_message, AIMessage):
        print("No valid AI message found, ending")
        return END
    
    response_content = (last_message.content or "").lower()
    print(f"Analyzing response: {response_content[:100]}...")
    
    # Check if the response indicates need for internet search
    if "should search internet" in response_content or "should search the internet" in response_content:
        print(" Search phrase detected! Going to search_node")
        return "search_node"
    else:
        print(" No search phrase found, ending")
        return END


def search_node(state: ChatbotState) -> dict:
    """Run Serper search and format top results."""
    print(" Starting internet search...")
    
    if not search:
        print(" No search API available")
        return {"search_results": None}

    # Get the original user query, not the LLM's response
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        user_message = state["messages"][-1].content if state["messages"] else ""
    
    print(f"Searching for: {user_message}")
    
    try:
        results = search.results(user_message)
        formatted = ""
        if isinstance(results, dict) and "organic" in results:
            for item in results["organic"][:3]:
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "")
                formatted += f"- {title}: {snippet} [Source: {link}]\n"
        if not formatted:
            formatted = "No relevant search results found."
        
        print(f"Found {len(results.get('organic', []))} results")
        print(f"Formatted results length: {len(formatted)} characters")
        return {"search_results": formatted}
    except Exception as e:
        print(f" Error: {e}")
        return {"search_results": f"Error fetching search results: {e}"}


def chatbot_node(state: ChatbotState) -> dict:
    """Generate the assistant reply using chat history, RAG context and optional search results."""
    print("Starting chatbot processing...")
    user_message = state["messages"][-1].content if state["messages"] else ""

    # Get RAG context and search results
    rag_context = state.get("rag_context", "")
    search_results = state.get("search_results") or ""

    # Determine if we have search results
    has_search_results = search_results and "No relevant search results found" not in search_results
    has_rag_context = rag_context and rag_context.strip() and len(rag_context.strip()) > 50

    if has_search_results:
        print("Using search results for final answer")
        # We have search results - provide comprehensive answer using them
        if has_rag_context:
            # Combine both RAG and search results
            system_instructions = (
                "You are a helpful assistant. Answer the question using the provided document context and search results. "
                "Use the document context as your primary source, but supplement with search results for additional information or current data. "
                "Provide a comprehensive answer based on all available information. Be confident in your response when you have good sources. "
                "Cite sources when using search results."
            )
            context_message = f"Document Context: {rag_context}\n\nSearch Results: {search_results}"
            print(f"Using both RAG context ({len(rag_context)} chars) and search results")
        else:
            # Use search results only
            system_instructions = (
                "You are a helpful assistant. Answer the question using the provided search results and conversation history. "
                "Use the search results to provide a comprehensive and informative answer. Be confident in your response when you have good search data. "
                "Reference previous parts of the conversation when relevant. Cite sources inline when using search results."
            )
            context_message = f"Search Results: {search_results}"
            print(f"Using search results only")

    else:
        print("First attempt - using RAG context and general knowledge")
        # First attempt - use RAG context and general knowledge
        system_instructions = (
            "You are a helpful assistant. Answer the question using the provided document context and conversation history. "
            "If the document context is not sufficient or irrelevant to answer the question, use your general knowledge. "
            "If you cannot answer the question with either the document context or your general knowledge, "
            "reply with exactly this phrase: 'I cannot answer this question with the available information. Should search internet.'"
        )
        
        if has_rag_context:
            context_message = f"Document Context: {rag_context}"
            print(f"Using RAG context ({len(rag_context)} chars)")
        else:
            context_message = "Document Context: No relevant documents found."
            print("No RAG context available, relying on general knowledge")

    # Build messages with conversation history
    messages = [SystemMessage(content=system_instructions)]
    
    # Add conversation history (all previous messages)
    messages.extend(state["messages"])
    
    # Add context information as a separate message
    messages.append(HumanMessage(content=f"Additional Context: {context_message}"))

    print(f"Sending {len(messages)} messages to LLM")
    try:
        response = llm.invoke(messages)
        print(f"Received response: {response.content[:100]}...")
        
        # Only check for search trigger if we don't already have search results
        if not has_search_results:
            response_content = (response.content or "").lower()
            should_search = "should search internet" in response_content or "should search the internet" in response_content
            
            if should_search:
                print("Response indicates search needed - setting requires_search=True")
            else:
                print("§Response complete - no search needed")
        else:
            should_search = False
            print("Final answer provided using search results")
        
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "requires_search": should_search and search is not None,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error generating response: {e}")],
            "requires_search": False,
        }


def start_node(state: ChatbotState) -> dict:
    print("Starting new conversation turn")
    return {"requires_search": False}


# Build LangGraph
workflow = StateGraph(ChatbotState)
workflow.add_node("start_node", start_node)
workflow.add_node("search_node", search_node)
workflow.add_node("chatbot_node", chatbot_node)

workflow.set_entry_point("start_node")
workflow.add_conditional_edges(
    "start_node", should_search, {"chatbot_node": "chatbot_node"}
)
workflow.add_conditional_edges(
    "chatbot_node", check_if_search_needed, {"search_node": "search_node", END: END}
)
workflow.add_edge("search_node", "chatbot_node")

app = workflow.compile()


def run_chatbot_with_logs():
    """Simple CLI loop for chatting."""
    state: ChatbotState = {
        "messages": [], 
        "search_results": None, 
        "rag_context": None,
        "requires_search": False
    }
    print("Chatbot started. Type 'quit' to exit.")
    print("=" * 50)
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        
        print("=" * 50)
        print(f"Question: {user_input}")
        
        state["messages"].append(HumanMessage(content=user_input))
        state["rag_context"] = None  # Reset for new query
        state["requires_search"] = False  # Reset for new query
        state["search_results"] = None  # Reset search results
        
        state = app.invoke(state)
        print("=" * 50)
        print("Bot:", state["messages"][-1].content)
        print("=" * 50)


if __name__ == "__main__":
    run_chatbot_with_logs()
