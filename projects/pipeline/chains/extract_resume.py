import json

from openai import OpenAI


def extract_resume(client: OpenAI, resume_text: str) -> dict:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "Extract structured fields from a resume. "
            "Return JSON only with keys: candidate_name, years_experience, skills, location_preference."
        ),
        input=(
            "Return valid JSON only.\n"
            f"Resume:\n{resume_text}"
        ),
        text={"format": {"type": "json_object"}},
        max_output_tokens=250,
    )
    return json.loads(response.output_text)
