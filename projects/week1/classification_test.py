from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

client = OpenAI(api_key=api_key)

class ClassificationTest(BaseModel):
    role_family: Literal["backend", "frontend", "full_stack", "data", "ai_ml", "other"]
    match_level: Literal["strong_match", "possible_match", "not_match"]
    recommended_action: Literal["apply_now", "review_later", "skip"]
    reason: str

job_samples = [
    """
    New Grad Software Engineer - Backend
    Location: Seattle, WA
    Requirements: Python, REST APIs, SQL, distributed systems.
    We welcome recent graduates.
    """,
    """
    Senior iOS Engineer
    Location: San Francisco, CA
    Requirements: 8+ years of Swift, iOS architecture, mobile performance.
    """,
    """
    AI Application Engineer
    Location: Remote US
    Requirements: Python, OpenAI API, prompt engineering, backend APIs, SQL.
    Entry-level candidates are welcome.
    """,
]
 
for index, job_sample in enumerate(job_samples, start=1):
    response = client.responses.parse(
        model="gpt-5.2",
        instructions=""" 
        You are a job matching assistant for an entry-level software engineer.

The candidate:
- is a CS master's student
- is looking for entry-level or new grad software engineer roles
- is interested in backend systems and AI applications
- prefers Seattle, Bellevue, Redmond, Kirkland, or Remote US
- has experience with Python, Java, SQL, APIs, gRPC, and basic AI API usage

Classify the job posting.
Be practical and concise.
""",
        input=job_sample,
        text_format=ClassificationTest,
    )
    result = response.output_parsed


    print("=" * 50)
    print(f"Job {index}")
    print(f"Role Family: {result.role_family}")
    print(f"Match Level: {result.match_level}")
    print(f"Recommended Action: {result.recommended_action}")
    print(f"Reason: {result.reason}")
