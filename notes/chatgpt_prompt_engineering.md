# ChatGPT Prompt Engineering for Developers

**Platform:** DeepLearning.AI
**Instructors:** Andrew Ng & Isa Fulford (OpenAI)
**Completed:** 2026-04-28

---

## 1. Base LLM vs Instruction-Tuned LLM

- **Base LLM**: predicts next token from training data — unpredictable for tasks
- **Instruction-tuned LLM**: fine-tuned + RLHF to follow instructions — this is what GPT-4o, Claude, etc. are
- Always think of the model as an assistant that follows explicit instructions — output quality is bounded by instruction quality

---

## 2. Two Core Principles of Prompting

### Principle 1: Write clear and specific instructions

| Tactic | Purpose | Example |
|---|---|---|
| Use delimiters | Isolate input, prevent prompt injection | ` ``` `, `"""`, `<text></text>` |
| Ask for structured output | Get JSON/HTML instead of prose | `"Return JSON with keys: title, score, reason"` |
| Check conditions first | Handle edge cases explicitly | `"If no job requirements found, return null"` |
| Few-shot examples | Show the pattern before asking | Give 1–2 input/output examples in the prompt |

### Principle 2: Give the model time to think

| Tactic | Purpose |
|---|---|
| Specify steps | `"Step 1: extract… Step 2: evaluate… Step 3: return JSON"` |
| Reason before concluding | `"Work out your answer before stating the final result"` — reduces wrong confident answers |

---

## 3. Iterative Prompt Development

The prompt is never perfect on the first try:

```
Write prompt → See output → Identify gap → Refine → Repeat
```

- Start simple, add constraints only when you see a real failure
- Test on multiple inputs before settling on a version
- Track prompt versions in your repo

---

## 4. Core Task Types

| Task | Key Pattern |
|---|---|
| Summarizing | `"Summarize in under 50 words, focus on X"` |
| Inferring | `"What is the sentiment? What topics are covered?"` |
| Transforming | `"Translate to French"` / `"Convert to JSON"` |
| Expanding | `"Write a professional reply to this email"` |

---

## 5. The Messages Format

```python
messages = [
    {"role": "system",   "content": "You are a helpful assistant."},
    {"role": "user",     "content": "What is RAG?"},
    {"role": "assistant","content": "RAG stands for..."},
    {"role": "user",     "content": "Give me an example."},
]
```

- **system**: model's persona and rules — runs once, shapes all responses
- **user**: what you send
- **assistant**: prior model replies — include these for conversation memory
- Without prior turns, the model has no memory of the conversation

---

## 6. Temperature

| Value | Behavior | Use case |
|---|---|---|
| `0` | Deterministic, same output every run | Structured extraction, evals, scoring |
| `0.7–1.0` | Creative, varied output | Brainstorming, writing |

---

## Key Takeaways

1. **Structured output is the foundation of AI application engineering.** JSON output is composable; prose is a dead end in a pipeline.
2. **Delimiters prevent prompt injection.** Always wrap user content: `<job_description>{jd}</job_description>`.
3. **The messages format IS the API.** Everything else — LangChain, agents, memory — builds on top of this.
4. **Use `temperature=0` for your job evaluation pipeline** — you want consistent, comparable scores.
