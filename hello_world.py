from typing import TypedDict
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

class ChatBotState(TypedDict):
    message: str 
    name: str

def complement_node(state: ChatBotState) -> ChatBotState:
    # Here you can add logic to modify the state or perform actions
    state["message"] = "Hi "+state["name"]+", You look beautiful today"
    return state  

workflow = StateGraph(ChatBotState)

workflow.add_node("complement", complement_node)

workflow.set_entry_point("complement")
workflow.set_finish_point("complement")

app = workflow.compile()

# display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"name": "Ludee"})
print(result["message"])  # Should print: Hi Ludee, You look beautiful today

