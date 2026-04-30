# Building Systems with the ChatGPT API

**Platform:** DeepLearning.AI
**API Used:** OpenAI Responses API (modern)
**Status:** In progress

---

## Course Outline

- [x] Lesson 1: OpenAI Responses API Basics
- [x] Lesson 2: Classification
- [x] Lesson 3: Moderation
- [x] Lesson 4: Chain of Thought
- [x] Lesson 5: Chaining Prompts
- [ ] Lesson 6: Check Outputs
- [ ] Lesson 7: Evaluation

---

## Lesson 1: OpenAI Responses API Basics

### Client setup

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()          # reads .env file — must be called before OpenAI()
client = OpenAI()      # reads OPENAI_API_KEY from environment automatically
```

### Making a call

```python
response = client.responses.create(
    model="gpt-4o",
    instructions="You are a helpful assistant.",   # model's role and rules
    input="Explain REST vs gRPC."                  # user's current request
)
print(response.output_text)   # easiest way to read the answer
```

### Three ways to write `input`

```python
# 1. Simple string — one message, no role needed
input="Explain what an API is."

# 2. Message list — conversation history or multi-role
input=[{"role": "user", "content": "Explain REST vs gRPC."}]

# 3. Content-part list — multimodal (text + image/file)
input=[{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "..."}}
]}]
```

### API key security

- Never hardcode the key in source code
- Store in `.env` file, load with `load_dotenv()`
- Always add `.env` to `.gitignore`

### Modern control knobs (not `temperature`)

| Knob | Purpose |
|---|---|
| `reasoning.effort` | How hard the model thinks: `"low"`, `"medium"`, `"high"` |
| `text.verbosity` | How detailed the response is |
| Structured Outputs | Force JSON schema compliance |

---

## Lesson 2: Classification

**Classification** = routing user input into predefined categories so your program knows which workflow to run next.

```python
def classify_user_input(user_message: str) -> dict:
    response = client.responses.create(
        model="gpt-4o",
        instructions="""
        Classify the user message for a job search assistant.
        Categories: job_search, resume_help, interview_prep, other

        Return JSON only:
        {
            "category": "...",
            "also_related_to": "... or null",
            "confidence": "high/medium/low",
            "clarification_needed": true/false
        }
        """,
        input=user_message
    )
    return json.loads(response.output_text)
```

### Handling edge cases

| Case | Action |
|---|---|
| `confidence: "low"` | Ask the user to clarify |
| `category: "other"` | Return "I can only help with job search topics" |
| Two categories at once | Set primary `category`, put secondary in `also_related_to` |

---

## Lesson 3: Moderation

A safety gate **before** the classifier — checks if input is harmful before the pipeline does anything.

```python
def is_input_safe(user_message: str) -> bool:
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=user_message
    )
    return not response.results[0].flagged
```

- `flagged=True` → harmful content detected, stop the pipeline
- Uses a dedicated safety model — more reliable than a general prompt
- **Free to call** — no token cost

### Pipeline position

```
user input → [moderation] → [classification] → [workflow]
                  ↓
              flagged?
                  ↓
           return safe reply, stop
```

---

## Lesson 4: Chain of Thought

Forces the model to reason step by step before concluding — improves accuracy for complex tasks like job fit evaluation.

```python
response = client.responses.create(
    model="gpt-4o",
    instructions="""
    Evaluate job fit using this process:
    Step 1: List required skills from the job description.
    Step 2: List the candidate's matching skills.
    Step 3: List the gaps.
    Step 4: Based on steps 1-3, give a fit score from 0-10.

    Return JSON:
    {
        "reasoning": "step by step analysis here",
        "fit_score": 0-10,
        "strengths": [],
        "gaps": []
    }
    """,
    input=f"Job: {job_description}\n\nResume: {resume}"
)
```

### Key rule

**Always include `reasoning` as a field inside the JSON** — never let the model write free text before the JSON block, or `json.loads()` will break.

### When to use it

- Complex comparisons (job fit evaluation) → yes
- Simple routing (classification) → not needed

---

## Lesson 5: Chaining Prompts

Break a complex task into a sequence of focused steps. Each step's output becomes the next step's input.

```python
def job_match_pipeline(job_description: str, resume: str) -> str:

    # Step 1: extract JD requirements
    step1 = client.responses.create(
        model="gpt-4o",
        instructions="Extract key requirements. Return JSON: {\"required_skills\": [], \"experience_years\": 0}",
        input=job_description
    )
    jd_data = json.loads(step1.output_text)

    # Step 2: extract candidate profile
    step2 = client.responses.create(
        model="gpt-4o",
        instructions="Extract candidate skills and experience. Return JSON: {\"skills\": [], \"experience_years\": 0}",
        input=resume
    )
    resume_data = json.loads(step2.output_text)

    # Step 3: evaluate fit with chain of thought
    step3 = client.responses.create(
        model="gpt-4o",
        instructions="""
        Compare job requirements and candidate profile.
        Step 1: list matching skills
        Step 2: list gaps
        Step 3: score the fit
        Return JSON: {"reasoning": "...", "fit_score": 0-10, "strengths": [], "gaps": []}
        """,
        input=f"Job: {jd_data}\n\nCandidate: {resume_data}"
    )
    fit_data = json.loads(step3.output_text)

    # Step 4: generate user-facing explanation
    step4 = client.responses.create(
        model="gpt-4o",
        instructions="Write a clear, encouraging job fit summary for the candidate.",
        input=f"Evaluation: {fit_data}"
    )
    return step4.output_text   # natural language to user
```

### Two rules for clean chains

1. **Pass structured JSON between steps, not raw text** — keeps each step focused
2. **Natural language only at the final step** — all intermediate steps return JSON

### Always handle JSON errors

```python
try:
    data = json.loads(response.output_text)
except json.JSONDecodeError:
    return "Sorry, something went wrong. Please try again."
```

---

## Full Pipeline Pattern

```
user input
    ↓
[moderation]      — is it safe?
    ↓
[classification]  — what does the user want?
    ↓
[chain step 1]    — extract structured data
    ↓
[chain step 2]    — extract more structured data
    ↓
[chain step 3]    — reason + evaluate (chain of thought)
    ↓
[chain step 4]    — natural language to user
```

This is the skeleton of the job-search assistant.
