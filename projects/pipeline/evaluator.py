import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.chains.extract_jd import extract_jd
from pipeline.chains.extract_resume import extract_resume
from pipeline.chains.fit_score import fit_score
from pipeline.classifier import classify_user_intent
from pipeline.moderation import moderate_text


load_dotenv()


def _build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")
    return OpenAI(api_key=api_key)


def _quality_check(output: dict[str, Any]) -> dict[str, Any]:
    required_keys = {"fit_score", "strengths", "gaps", "recommendation", "rationale"}
    missing = sorted(required_keys - set(output.keys()))
    fit_score_ok = isinstance(output.get("fit_score"), int) and 0 <= output["fit_score"] <= 10
    recommendation_ok = output.get("recommendation") in {"apply_now", "review_later", "skip"}
    strengths_ok = isinstance(output.get("strengths"), list)
    gaps_ok = isinstance(output.get("gaps"), list)
    rationale_ok = isinstance(output.get("rationale"), str) and len(output["rationale"].strip()) > 0

    passed = (
        not missing
        and fit_score_ok
        and recommendation_ok
        and strengths_ok
        and gaps_ok
        and rationale_ok
    )
    return {
        "passed": passed,
        "missing_keys": missing,
        "fit_score_ok": fit_score_ok,
        "recommendation_ok": recommendation_ok,
        "strengths_ok": strengths_ok,
        "gaps_ok": gaps_ok,
        "rationale_ok": rationale_ok,
    }


def run_pipeline(user_input: str, job_description: str, resume_text: str) -> dict:
    client = _build_client()

    # 1) input moderation
    in_moderation = moderate_text(client, user_input, stage="input")
    if in_moderation["flagged"]:
        return {
            "status": "blocked_input",
            "moderation": in_moderation,
        }

    # 2) classification
    route = classify_user_intent(client, user_input)
    if route["route"] == "reject":
        return {"status": "rejected_by_classifier", "route": route}
    if route["route"] == "ask_clarification":
        return {"status": "need_clarification", "route": route}

    # 3) chain steps
    jd_info = extract_jd(client, job_description)
    resume_info = extract_resume(client, resume_text)
    fit = fit_score(client, jd_info, resume_info)

    # 4) output moderation
    output_text = str(fit)
    out_moderation = moderate_text(client, output_text, stage="output")
    if out_moderation["flagged"]:
        return {
            "status": "blocked_output",
            "route": route,
            "jd_info": jd_info,
            "resume_info": resume_info,
            "raw_output": fit,
            "moderation": out_moderation,
        }

    # 5) quality check
    quality = _quality_check(fit)
    if not quality["passed"]:
        return {
            "status": "quality_failed",
            "route": route,
            "jd_info": jd_info,
            "resume_info": resume_info,
            "output": fit,
            "quality": quality,
        }

    # 6) output
    return {
        "status": "ok",
        "route": route,
        "jd_info": jd_info,
        "resume_info": resume_info,
        "quality": quality,
        "output": fit,
    }
