from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, TypedDict
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure it is set correctly.")

# Define the chatbot state using TypedDict for better type safety
class ChatbotState(TypedDict):
    messages: List[HumanMessage | AIMessage]

# Initialize the Gemini model with explicit API key
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=gemini_api_key
    )
except Exception as e:
    raise Exception(f"Failed to initialize Gemini model: {str(e)}")

# Define the chatbot node
def chatbot_node(state: ChatbotState) -> dict:
    # Get the latest user message
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # Create a prompt with conversation history
    messages = state["messages"] + [HumanMessage(content=user_message)]
    
    # Generate response using Gemini
    try:
        response = llm.invoke(messages)
        # Return updated messages list as a dictionary
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}
    except Exception as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error: Failed to generate response: {str(e)}")]}

# Set up the LangGraph workflow
workflow = StateGraph(ChatbotState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Compile the graph
app = workflow.compile()

# Interactive loop for testing
if __name__ == "__main__":
    state = {"messages": []}
    print("Chatbot started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        # Run the graph
        state = app.invoke(state)
        # Print the latest AI response
        print("Bot:", state["messages"][-1].content)