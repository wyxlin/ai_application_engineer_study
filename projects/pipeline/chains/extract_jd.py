import json

from openai import OpenAI


def extract_jd(client: OpenAI, job_description: str) -> dict:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "Extract structured fields from a job description. "
            "Return JSON only with keys: role_title, seniority, required_skills, preferred_location."
        ),
        input=(
            "Return valid JSON only.\n"
            f"Job Description:\n{job_description}"
        ),
        text={"format": {"type": "json_object"}},
        max_output_tokens=250,
    )
    return json.loads(response.output_text)
