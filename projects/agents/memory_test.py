from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define tools
@tool
def search_jobs(role: str, location: str) -> str:
    """Search for job listings by role and location."""
    role = role.lower()
    location = location.lower()
    if "backend" in role and "seattle" in location:
        return "\n".join([
            "Backend Engineer at Amazon: Python, distributed systems, AWS. Salary: $180k",
            "Backend Engineer at Microsoft: Python, REST APIs, Azure. Salary: $170k",
        ])
    elif "frontend" in role and "seattle" in location:
        return "Frontend Engineer at Expedia: React, TypeScript, GraphQL"
    return "No jobs found for this search."

@tool
def get_resume() -> str:
    """Get the candidate's resume."""
    return "Jason Yuan: Python, gRPC, distributed systems, SQL, backend engineering"

# Define agent with memory
memory = MemorySaver()
tools = [search_jobs, get_resume]   
agent = create_react_agent(llm, tools=tools, checkpointer=memory)

# Same thread_id = same conversation
config = {"configurable": {"thread_id": "session_001"}}

print("=== Turn 1 ===")
response = agent.invoke(
    {"messages": [("user", "Find backend jobs in Seattle")]},
    config
)

print(response["messages"][-1].content)

print("\n=== Turn 2 ===")
response = agent.invoke(
    {"messages": [("user", "which one pays more?")]},
    config
)
print(response["messages"][-1].content)

print("\n=== Turn 3 ===")
response = agent.invoke(
    {"messages": [("user", "Which one better fits my resume?")]},
    config
)
print(response["messages"][-1].content)

