
from src.config.config import Config
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def calculator(query: str) -> str:
    """Return a simple test result."""
    return "Tool executed successfully."


# Initialize Ollama
llm = Config.get_llm()

# Create ReAct agent
agent = create_react_agent(
    llm,
    tools=[calculator]
)

# Test tool calling
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Use the calculator tool to test the tool calling capability."
        }
    ]
})

# Display results
for message in result["messages"]:
    print(f"\nRole: {message.type}")
    print(f"Content: {message.content}")

    if message.type == "ai" and message.tool_calls:
        print("Tool calls:", message.tool_calls)
