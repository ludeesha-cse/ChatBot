from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import List, TypedDict, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure it is set correctly.")
if not serper_api_key:
    raise ValueError("SERPER_API_KEY not found in .env file. Please obtain it from https://serper.dev.")

# Define the chatbot state
class ChatbotState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    search_results: Optional[str]
    needs_search: bool

# Initialize the Gemini model
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        google_api_key=gemini_api_key
    )
except Exception as e:
    raise Exception(f"Failed to initialize Gemini model: {str(e)}")

# Initialize the Serper API wrapper
try:
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
except Exception as e:
    raise Exception(f"Failed to initialize Serper API: {str(e)}")

# Decision function to determine if search is needed using Gemini
def should_search(state: ChatbotState) -> str:
    user_message = state["messages"][-1].content if state["messages"] else ""
    classification_prompt = (
        f"Determine if the following user query requires a web search to provide an accurate, up-to-date response. "
        f"Return 'yes' if the query needs real-time data (e.g., news, weather, recent events) or specific external information. "
        f"Return 'no' if the query can be answered with general knowledge or conversation. "
        f"Query: {user_message}"
    )
    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        decision = response.content.strip().lower()
        return "search_node" if decision == "yes" else "chatbot_node"
    except Exception as e:
        return "chatbot_node"  # Default to chatbot if classification fails

# Search node to fetch and format web results
def search_node(state: ChatbotState) -> dict:
    user_message = state["messages"][-1].content if state["messages"] else ""
    try:
        # Use Serper API to get structured results
        results = search.results(user_message)  # Use .results() for raw JSON
        formatted_results = ""
        if "organic" in results:
            for item in results["organic"][:3]:  # Limit to top 3 organic results
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "No link")
                formatted_results += f"- {title}: {snippet} [Source: {link}]\n"
        if not formatted_results:
            formatted_results = "No relevant search results found."
        return {"search_results": formatted_results, "needs_search": False}
    except Exception as e:
        return {"search_results": f"Error: Failed to fetch search results: {str(e)}", "needs_search": False}

# Chatbot node to generate response
def chatbot_node(state: ChatbotState) -> dict:
    user_message = state["messages"][-1].content if state["messages"] else ""
    search_results = state.get("search_results", "")
    
    # Create system message with instructions
    system_instructions = (
        "You are a helpful assistant. Use the conversation history to maintain context and provide relevant responses. "
        "Reference previous parts of the conversation when relevant. "
    )
    
    # Build messages with conversation history
    messages = [HumanMessage(content=system_instructions)]
    
    # Add all conversation history
    messages.extend(state["messages"])
    
    # Add search results as additional context if available
    if search_results and "No relevant search results found" not in search_results:
        messages.append(HumanMessage(content=f"Additional Context from Web Search: {search_results}\n\nPlease provide a response using both the conversation history and search results. Include source links when referencing search results."))
    
    try:
        response = llm.invoke(messages)
        return {"messages": state["messages"] + [AIMessage(content=response.content)], "needs_search": False}
    except Exception as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error: Failed to generate response: {str(e)}")], "needs_search": False}

# Start node to initialize processing
def start_node(state: ChatbotState) -> dict:
    return {"needs_search": True}

# Set up the LangGraph workflow
workflow = StateGraph(ChatbotState)
workflow.add_node("start_node", start_node)
workflow.add_node("search_node", search_node)
workflow.add_node("chatbot_node", chatbot_node)

# Define edges
workflow.set_entry_point("start_node")
workflow.add_conditional_edges(
    "start_node",
    should_search,
    {"search_node": "search_node", "chatbot_node": "chatbot_node"}
)
workflow.add_edge("search_node", "chatbot_node")
workflow.add_edge("chatbot_node", END)

# Compile the graph
app = workflow.compile()

# Interactive loop for testing
if __name__ == "__main__":
    state = {"messages": [], "search_results": None, "needs_search": True}
    print("Chatbot started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        state["messages"].append(HumanMessage(content=user_input))
        state["needs_search"] = True
        state = app.invoke(state)
        print("Bot:", state["messages"][-1].content)