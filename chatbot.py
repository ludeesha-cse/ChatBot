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
    """Ask the LLM if we need a web search for the last user message."""
    if not search:
        return "chatbot_node"

    user_message = state["messages"][-1].content if state["messages"] else ""
    classification_prompt = (
        "You are a classifier that decides if a user query requires a web search for fresh/external facts.\n"
        "Return exactly one word: yes or no.\n\n"
        f"Query: {user_message}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=classification_prompt)])
        decision = (resp.content or "").strip().lower()
        return "search_node" if decision.startswith("y") else "chatbot_node"
    except Exception:
        return "chatbot_node"


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

    # Retrieve RAG context from vector DB
    rag_context = ""
    try:
        rag_context = retrieve_context(user_message, k=3)
    except Exception as e:
        rag_context = f"RAG retrieval error: {e}"

    search_results = state.get("search_results") or ""

    system_instructions = (
        "You are a helpful assistant. Use provided context snippets when relevant.\n"
        "If you use search results, cite the sources inline.\n"
        "If the context is insufficient, say so briefly.\n"
    )

    # Build conversation with history and enrich the last user turn with RAG/search context
    history: List[HumanMessage | AIMessage] = list(state.get("messages", []))
    if history and isinstance(history[-1], HumanMessage):
        last_user: HumanMessage = history.pop()  # remove last user to augment it
        enriched_last = HumanMessage(
            content=(
                f"{last_user.content}\n\n"
                f"Context (from documents):\n{rag_context}\n\n"
                + (f"Search Results:\n{search_results}\n\n" if search_results else "")
                + "Answer concisely and accurately; include numbered sources if used."
            )
        )
        convo_messages = [SystemMessage(content=system_instructions), *history, enriched_last]
    else:
        convo_messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_message),
        ]

    try:
        response = llm.invoke(convo_messages)
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
    state: ChatbotState = {"messages": [], "search_results": None, "needs_search": True}
    print("Chatbot started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        state["messages"].append(HumanMessage(content=user_input))
        state["needs_search"] = True
        state = app.invoke(state)
        print("Bot:", state["messages"][-1].content)


if __name__ == "__main__":
    run_chatbot()
