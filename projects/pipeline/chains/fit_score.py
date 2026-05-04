import json

from openai import OpenAI


def fit_score(client: OpenAI, jd_info: dict, resume_info: dict) -> dict:
    response = client.responses.create(
        model="gpt-5.2",
        instructions=(
            "You are a strict job-fit evaluator. "
            "Think through the comparison internally and return only final JSON. "
            "JSON keys: fit_score (0-10 int), strengths (array), gaps (array), recommendation (apply_now|review_later|skip), rationale (string)."
        ),
        input=(
            "Return valid JSON only.\n"
            f"JD_INFO: {json.dumps(jd_info, ensure_ascii=False)}\n"
            f"RESUME_INFO: {json.dumps(resume_info, ensure_ascii=False)}"
        ),
        text={"format": {"type": "json_object"}},
        max_output_tokens=400,
    )
    return json.loads(response.output_text)
