import json

from openai import OpenAI


def classify_user_intent(client: OpenAI, user_input: str) -> dict:
    """
    Route user input to one of:
    - evaluate_job_fit
    - ask_clarification
    - reject
    """
    lowered = user_input.lower()

    # Fast-path heuristic: common evaluation intents should route directly.
    eval_keywords = [
        "evaluate",
        "fit",
        "match",
        "score",
        "rate",
        "apply",
        "should this candidate",
        "job-fit",
        "job fit",
        "resume",
        "jd",
    ]
    if any(keyword in lowered for keyword in eval_keywords):
        return {
            "route": "evaluate_job_fit",
            "reason": "Matched evaluation intent keywords.",
        }

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You are a routing classifier for a job-matching pipeline. "
            "Return JSON only with keys: route, reason. "
            "route must be one of evaluate_job_fit, ask_clarification, reject. "
            "Default to evaluate_job_fit when the user asks to compare a resume/profile/candidate "
            "with a job/JD/role, asks for fit/match/score/rating/recommendation/apply decision. "
            "Use ask_clarification only when the request intent itself is unclear. "
            "Do not use ask_clarification just because details are missing."
        ),
        input=(
            "Classify this user input. Return valid JSON.\n"
            f"User input:\n{user_input}"
        ),
        text={"format": {"type": "json_object"}},
        max_output_tokens=120,
    )

    parsed = json.loads(response.output_text)
    route = parsed.get("route", "ask_clarification")
    if route not in {"evaluate_job_fit", "ask_clarification", "reject"}:
        route = "ask_clarification"
    return {"route": route, "reason": parsed.get("reason", "")}
