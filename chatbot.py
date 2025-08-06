from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure it is set correctly.")

# Define the chatbot state
class ChatbotState:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

# Initialize the Gemini model with explicit API key
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        google_api_key=gemini_api_key
    )
except Exception as e:
    raise Exception(f"Failed to initialize Gemini model: {str(e)}")

# Define the chatbot node
def chatbot_node(state: ChatbotState) -> ChatbotState:
    # Get the latest user message
    user_message = state.messages[-1].content if state.messages else ""
    
    # Create a prompt with conversation history
    messages = state.messages + [HumanMessage(content=user_message)]
    
    # Generate response using Gemini
    try:
        response = llm.invoke(messages)
        # Update state with the new AI response
        state.messages.append(AIMessage(content=response.content))
    except Exception as e:
        state.messages.append(AIMessage(content=f"Error: Failed to generate response: {str(e)}"))
    
    return state

# Temporary test script
if __name__ == "__main__":
    state = ChatbotState()
    state.messages = [HumanMessage(content="Hello, how are you?")]
    state = chatbot_node(state)
    print(state.messages[-1].content)  # Print the AI's response