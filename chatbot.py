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
    search_attempted: bool


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


def retrieve_rag_context_node(state: ChatbotState) -> dict:
    """Retrieve RAG context and store it in state."""
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # Retrieve RAG context
    try:
        rag_context = retrieve_context(user_message, k=3)
        retrieved_context = rag_context if rag_context else ""
    except Exception as e:
        retrieved_context = ""
    
    return {"rag_context": retrieved_context}


def should_search(state: ChatbotState) -> str:
    """Always go to chatbot first - retrieve RAG context and let LLM decide."""
    return "retrieve_rag_context_node"


def check_if_search_needed(state: ChatbotState) -> str:
    """Check if the LLM's response indicates it needs search."""
    
    if not search:
        return END
        
    if not state.get("requires_search", False):
        return END
    
    # If we already attempted search, don't search again
    if state.get("search_attempted", False):
        return END
    
    # Get the last AI response
    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message or not isinstance(last_message, AIMessage):
        return END
    
    response_content = (last_message.content or "").lower()
    
    # Check if the response indicates need for internet search
    if "should search internet" in response_content or "should search the internet" in response_content:
        return "search_node"
    else:
        return END


def search_node(state: ChatbotState) -> dict:
    """Run Serper search and format top results."""
    
    if not search:
        return {"search_results": None, "search_attempted": True}

    # Get the original user query, not the LLM's response
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        user_message = state["messages"][-1].content if state["messages"] else ""
        
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
        
        return {"search_results": formatted, "search_attempted": True}
    except Exception as e:
        return {"search_results": f"Error fetching search results: {e}", "search_attempted": True}


def chatbot_node(state: ChatbotState) -> dict:
    """Generate the assistant reply using chat history, RAG context and optional search results."""
    user_message = state["messages"][-1].content if state["messages"] else ""

    # Get RAG context and search results
    rag_context = state.get("rag_context", "")
    search_results = state.get("search_results") or ""
    search_attempted = state.get("search_attempted", False)

    # Determine if we have search results
    has_search_results = search_results and "No relevant search results found" not in search_results
    
    # Check if we have valid RAG context (not error messages or "No relevant documents found")
    has_rag_context = (rag_context and 
                      rag_context.strip() and 
                      len(rag_context.strip()) > 50 and
                      not rag_context.startswith("No relevant documents found") and
                      not rag_context.startswith("Retrieval error:"))

    # Check if this is a question or statement
    is_question = any(word in user_message.lower() for word in ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 'could you', 'would you', 'will you', 'do you', 'are you', 'is it', 'tell me', 'answer me'])

    if has_search_results:
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
        else:
            # Use search results only
            system_instructions = (
                "You are a helpful assistant. Answer the question using the provided search results and conversation history. "
                "Use the search results to provide a comprehensive and informative answer. Be confident in your response when you have good search data. "
                "Reference previous parts of the conversation when relevant. Cite sources inline when using search results."
            )
            context_message = f"Search Results: {search_results}"

    else:
        # First attempt or after failed search - use RAG context and general knowledge
        if search_attempted or not is_question:
            # If search was already attempted and failed, OR if it's not a question, just use general knowledge
            system_instructions = (
                "If the user is sharing information about themselves, acknowledge it politely and remember it for future conversation. "
                "If the user is asking a question, use the provided document context if relevant, otherwise use your general knowledge to provide a helpful response. "
                "Be conversational and friendly in your responses."
            )
        else:
            # First attempt with a question - allow search trigger
            system_instructions = (
                "Answer the question using the provided document context and conversation history. "
                "If the document context is not sufficient or irrelevant to answer the question, use your general knowledge. "
                "If you cannot answer the question with either the document context or your general knowledge, "
                "reply with exactly this phrase: 'I cannot answer this question with the available information. Should search internet.'"
            )
        
        if has_rag_context:
            context_message = f"Document Context: {rag_context}"
        else:
            context_message = "Document Context: No relevant documents found."

    # Build messages with conversation history
    messages = [SystemMessage(content=system_instructions)]
    
    # Add conversation history (all previous messages)
    messages.extend(state["messages"])
    
    # Add context information as a separate message
    messages.append(HumanMessage(content=f"Additional Context: {context_message}"))

    try:
        response = llm.invoke(messages)
        
        # Only check for search trigger if we don't already have search results
        if not has_search_results and not search_attempted and is_question:
            response_content = (response.content or "").lower()
            should_search = "should search internet" in response_content or "should search the internet" in response_content

        else:
            should_search = False
        
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "requires_search": should_search and search is not None,
        }
    except Exception as e:
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error generating response: {e}")],
            "requires_search": False,
        }


def start_node(state: ChatbotState) -> dict:
    return {"requires_search": False, "search_attempted": False}


# Build LangGraph
workflow = StateGraph(ChatbotState)
workflow.add_node("start_node", start_node)
workflow.add_node("retrieve_rag_context_node", retrieve_rag_context_node)
workflow.add_node("search_node", search_node)
workflow.add_node("chatbot_node", chatbot_node)

workflow.set_entry_point("start_node")
workflow.add_conditional_edges(
    "start_node", should_search, {"retrieve_rag_context_node": "retrieve_rag_context_node"}
)
workflow.add_edge("retrieve_rag_context_node", "chatbot_node")
workflow.add_conditional_edges(
    "chatbot_node", check_if_search_needed, {"search_node": "search_node", END: END}
)
workflow.add_edge("search_node", "chatbot_node")

app = workflow.compile()


def run_chatbot():
    """Simple CLI loop for chatting."""
    state: ChatbotState = {
        "messages": [], 
        "search_results": None, 
        "rag_context": None,
        "requires_search": False,
        "search_attempted": False
    }
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        state["rag_context"] = None  # Reset for new query
        state["requires_search"] = False  # Reset for new query
        state["search_attempted"] = False  # Reset search attempted flag
        state["search_results"] = None  # Reset search results
        state = app.invoke(state)
        print("Bot:", state["messages"][-1].content)


if __name__ == "__main__":
    run_chatbot()
