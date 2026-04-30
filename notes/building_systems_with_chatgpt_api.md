# Building Systems with the ChatGPT API

**Platform:** DeepLearning.AI
**API Used:** OpenAI Responses API (modern)
**Status:** Completed

---

## Course Outline

- [x] Lesson 1: OpenAI Responses API Basics
- [x] Lesson 2: Classification
- [x] Lesson 3: Moderation
- [x] Lesson 4: Chain of Thought
- [x] Lesson 5: Chaining Prompts
- [x] Lesson 6: Check Outputs
- [x] Lesson 7: Evaluation

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

---

## Lesson 6: Check Outputs

Mirror of Moderation — checks the model's output before sending it to the user.

### Safety check (free)

```python
def is_output_safe(model_response: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=model_response
    )
    return not result.results[0].flagged
```

### Quality check (costs tokens — use sparingly)

```python
def is_output_relevant(user_message: str, model_response: str) -> bool:
    check = client.responses.create(
        model="gpt-4o",
        instructions="""
        Does the response appropriately answer the user's message?
        Return JSON only: {"is_good": true/false, "reason": "..."}
        """,
        input=f"User message: {user_message}\n\nResponse: {model_response}"
    )
    result = json.loads(check.output_text)
    return result["is_good"]
```

### Rules

- Always run the moderation safety check on output — it's free
- Only run the quality check on the **final response** before it reaches the user, not on every intermediate step
- If output fails either check, return a fallback message instead

### Updated full pipeline

```
user input
    ↓
[input moderation]    — is input safe?
    ↓
[classification]      — what does the user want?
    ↓
[chain steps]         — structured JSON between steps
    ↓
[output moderation]   — is output safe?
    ↓
[quality check]       — is output relevant? (final step only)
    ↓
user
```

---

## Lesson 7: Evaluation

Systematically measuring how well your system performs — so you can catch regressions, compare prompt versions, and know your accuracy before an interview demo.

### Basic eval pattern

```python
test_cases = [
    {"input": "Find me a backend job in Seattle", "expected_category": "job_search"},
    {"input": "How can I improve my resume?",     "expected_category": "resume_help"},
    {"input": "What to say about my weakness?",   "expected_category": "interview_prep"},
]

def evaluate_classifier(test_cases: list) -> dict:
    correct = 0
    for case in test_cases:
        result = classify_user_input(case["input"])
        if result["category"] == case["expected_category"]:
            correct += 1
    return {"accuracy": correct / len(test_cases), "total": len(test_cases)}
```

### Two levels of evaluation

| Level | What it checks | Use case |
|---|---|---|
| **Exact match** | Output matches expected value exactly | Classification, scores |
| **LLM-graded** | Ask a model to judge quality | Natural language outputs |

### LLM-graded eval

```python
def grade_response(user_input: str, system_response: str, criteria: str) -> dict:
    result = client.responses.create(
        model="gpt-4o",
        instructions=f"Grade this response based on: {criteria}. Return JSON: {{\"score\": 1-5, \"reason\": \"...\"}}",
        input=f"User: {user_input}\nResponse: {system_response}"
    )
    return json.loads(result.output_text)
```

### Key rules

- Start collecting eval cases now — every manual test is a potential test case
- Before deploying a new prompt version, run your full eval set to catch regressions
- Target: 20–30 eval cases by Week 8 of the roadmap
