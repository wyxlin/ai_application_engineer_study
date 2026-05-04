from openai import OpenAI


def moderate_text(client: OpenAI, text: str, stage: str) -> dict:
    """
    stage: "input" or "output"
    """
    moderation = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    result = moderation.results[0]
    return {
        "stage": stage,
        "flagged": result.flagged,
        "categories": result.categories.model_dump(),
        "category_scores": result.category_scores.model_dump(),
    }
