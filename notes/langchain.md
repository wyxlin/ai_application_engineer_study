# LangChain for LLM Application Development

**Platform:** DeepLearning.AI
**Status:** In progress

---

## Course Outline

- [x] Prompt Templates
- [x] Models
- [x] Output Parsers
- [x] Chains
- [x] Retry and Error Handling
- [x] LangSmith (Logging and Tracing)

---

## Core Pattern: LCEL (LangChain Expression Language)

```python
chain = prompt | llm | parser
result = chain.invoke({"variable": "value"})
```

Each component's output feeds automatically into the next.

---

## Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a job fit evaluator."),
    ("user", "<job_description>{job_description}</job_description>\n\n<resume>{resume}</resume>")
])
```

**Load from file:**
```python
with open("prompts/job_evaluator.txt") as f:
    system_prompt = f.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])
```

---

## Model

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

**Switch to Anthropic — change 2 lines only:**
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-4-7", temperature=0)
```

Everything else — prompt, parser, chain — stays the same.

---

## Output Parsers

### JsonOutputParser — flexible, returns dict
```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class JobEvaluation(BaseModel):
    reasoning: str
    fit_score: int
    strengths: list[str]
    gaps: list[str]

parser = JsonOutputParser(pydantic_object=JobEvaluation)
chain = prompt | llm | parser
result = chain.invoke({...})
# result is a plain dict: result["fit_score"]
```

### with_structured_output() — strict, returns Pydantic object
```python
structured_llm = llm.with_structured_output(JobEvaluation)
chain = prompt | structured_llm
result = chain.invoke({...})
# result is a Pydantic object: result.fit_score
```

| | `JsonOutputParser` | `with_structured_output()` |
|---|---|---|
| Returns | Plain dict | Pydantic object |
| Schema enforcement | Lenient | Strict |
| Needs "return JSON" in prompt | Yes | No |

---

## Multi-Chain Pipeline

```python
# Define schemas
class JDRequirements(BaseModel):
    required_skills: list[str]
    experience_years: int

class ResumeProfile(BaseModel):
    skills: list[str]
    experience_years: int

class FitEvaluation(BaseModel):
    fit_score: int        # always specify range in prompt, not schema
    reasoning: str
    strengths: list[str]
    gaps: list[str]

# Define chains
jd_chain     = jd_prompt     | llm.with_structured_output(JDRequirements)
resume_chain = resume_prompt | llm.with_structured_output(ResumeProfile)
fit_chain    = fit_prompt    | llm.with_structured_output(FitEvaluation)

# Connect chains
def evaluate_pipeline(job_description: str, resume: str) -> FitEvaluation:
    jd_data     = jd_chain.invoke({"job_description": job_description})
    resume_data = resume_chain.invoke({"resume": resume})
    result      = fit_chain.invoke({
        "jd_requirements": jd_data,
        "resume_profile": resume_data
    })
    return result
```

---

## Retry and Error Handling

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)
fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

robust_chain = (
    prompt
    | llm.with_retry(stop_after_attempt=3)
         .with_structured_output(FitEvaluation)
         .with_fallbacks([
             fallback_llm.with_structured_output(FitEvaluation)
         ])
)
```

| Error type | Solution |
|---|---|
| API errors (rate limit, timeout) | `.with_retry()` |
| Output errors (bad data) | `.with_fallbacks()` |

---

## LangSmith — Observability

Add to `.env` — no code changes needed:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=job-search-assistant
```

LangSmith records for every chain run:
- Full formatted prompt sent to the model
- Model used, latency, token count, cost
- Input and output at each step

Use it to debug which step in a multi-chain pipeline produced wrong output.

---

## Key Rules

1. **Prompt and schema must agree** — model follows prompt instructions more than the schema. Name your JSON keys explicitly in the prompt.
2. **`with_structured_output()` enforces field types, not value ranges** — put range constraints in the prompt (`"score from 0 to 10"`).
3. **Separate prompts from code** — save as `.txt` files, load at runtime.
4. **`model_dump()`** converts a Pydantic object to a plain dict when needed downstream.
