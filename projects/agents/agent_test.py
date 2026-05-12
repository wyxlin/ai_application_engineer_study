from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define tools - functions the LLM can call
@tool
def search_jobs(role: str, location: str) -> str:
    """Search for job listings by role and location."""
    # Simulated results - later this calls Greenhouse/Lever API
    role = role.lower()
    location = location.lower()
    if "backend" in role and "seattle" in location:
        return "\n".join([
            "Backend Engineer at Amazon: Python, distributed systems, AWS",
            "Backend Engineer at Microsoft: Python, REST APIs, Azure",
        ])
    elif "frontend" in role and "seattle" in location:
        return "Frontend Engineer at Expedia: React, TypeScript, GraphQL"
    return "No jobs found for this search."

@tool
def evaluate_fit(job_description: str, resume: str) -> str:
    """Evaluate how well a candidate's resume fits a job description."""
    # Simulated evaluation — later this calls your LLM pipeline
    if "Python" in job_description and "Python" in resume:
        return f"Good fit (7/10): Candidate matches key requirements."
    return f"Partial fit (4/10): Some skills missing."

@tool
def get_resume() -> str:
    """Get the candidate's resume."""
    return "Jason Yuan: Python, gRPC, distributed systems, SQL, backend engineering"

# Guardrail in the system prompt
system_prompt = """ You are a job search assistant. You help candidate find and evaluate jobs.
Allowed actions:
- Search for jobs using search_jobs tool
- Evaluate job fit using evaluate_fit tool
- Get the candidate's resume using get_resume tool

Rules:
- Always retrieve the resume before evaluating fit
- Evaluate ALL jobs returned by search_jobs tool
- If no job are found, tell the user clearly - do not make up jobs
- Maximum 5 tool calls per response
-Always end with a clear recommendation
"""

# Create agent with tools
tools = [search_jobs, evaluate_fit, get_resume]
agent = create_react_agent(llm, tools=tools, prompt=system_prompt)

# Run agent
response = agent.invoke({
    "messages": [
        ("user", "Find backend jobs in Seattle and tell me which one I should apply to first.")
    ]
})

print(response["messages"][-1].content)

