# AI Application Engineering Roadmap

**Target roles:** AI Application Engineer · LLM Engineer · AI Software Engineer

**Duration:** 16 weeks — focus on building, not passive watching.

**Core strategy:** Learn a concept → immediately apply it to your job-search AI system → package the result for interviews.

---

## Portfolio Outcomes by Week 16

| Outcome | What It Demonstrates |
|---|---|
| AI Job Matching System | LLM decision pipeline, structured outputs, RAG, agents, evaluation, reliability |
| Resume + JD Evaluator | Small, interview-friendly AI app with clear inputs/outputs |
| Portfolio Document QA | RAG over your own resume, project docs, and READMEs |
| Interview Story | Ability to explain AI tradeoffs: prompts, RAG, agents, eval, cost, fallback |

---

## Month 1 — LLM Application Foundations

> Goal: Build a basic AI application using LLM APIs, structured outputs, and a clean project workflow.

### Week 1 — Prompt Engineering and Structured Output

**Learn:**
- LLM basics for application developers
- Prompt structure: task, context, constraints, examples
- JSON / schema-style structured outputs
- Basic OpenAI **and** Anthropic API calls in Python

**Deliverables:**
- `job_description → fit_score` JSON function using both OpenAI and Anthropic
- Prompt template v1 in your repo
- README note: input, output, limitations, and which provider you chose and why

**Resources:**
- ChatGPT Prompt Engineering for Developers — DeepLearning.AI
- OpenAI API docs + Structured Outputs cookbook
- Anthropic API docs (Messages API quickstart)

> **Added gap:** Use both OpenAI and Anthropic APIs for the same task. Understand the API surface differences. This shows provider breadth in interviews.

---

### Week 2 — LLM Systems and Multi-step Decision Pipelines

**Learn:**
- Chaining prompts and model calls
- Classification, routing, summarization, decision steps
- Separating deterministic logic from LLM reasoning
- **Streaming responses** — how to stream tokens for real-time UX

**Deliverables:**
- AI matching pipeline: extract → evaluate → decide → explain
- Structured output schema for job fit results
- At least 10 sample job evaluations saved as JSON
- One endpoint that streams the evaluation result token by token

**Resources:**
- Building Systems with the ChatGPT API — DeepLearning.AI
- OpenAI streaming guide
- Anthropic streaming docs

> **Added gap:** Streaming is ubiquitous in production. Every AI app you demo should stream. Add it here before building habits around blocking calls.

---

### Week 3 — LLM Application Engineering with LangChain

**Learn:**
- Chains and composable LLM workflows
- Prompt templates, output parsing, and validation
- Basic retry and error handling
- LangSmith for tracing and logging

**Deliverables:**
- Refactored LLM code into modules
- Retry and output validation added
- Logging for prompts, outputs, model, cost estimates
- LangSmith trace for at least one run

**Resources:**
- LangChain for LLM Application Development — DeepLearning.AI
- LangChain docs · LangSmith docs

---

### Week 4 — Mini Project: Resume and JD Evaluator

**Learn:**
- Building a small end-to-end AI product
- Designing user-facing AI output
- Explaining strengths, gaps, risks, suggestions

**Deliverables:**
- Resume + JD evaluator: score, strengths, gaps, suggestions
- Clean README with sample outputs
- Supports both blocking and streaming output modes

**Resources:**
- OpenAI Structured Outputs cookbook
- Anthropic prompt engineering docs

---

## Month 2 — Retrieval, Embeddings, and RAG

> Goal: Ground LLM outputs in external data and build retrieval-based AI applications.

### Week 5 — Embeddings and Vector Search

**Learn:**
- What embeddings represent
- Semantic similarity and vector search workflow
- When vector search is useful and when it is not
- **Model selection for embeddings**: cost vs. quality tradeoffs (OpenAI `text-embedding-3-small` vs `large`)

**Deliverables:**
- Embedding pipeline for resumes and job descriptions
- Similarity search demo
- Short note explaining how matching works and which embedding model you chose and why

**Resources:**
- Vector Databases: from Embeddings to Applications — DeepLearning.AI
- FAISS GitHub repo
- OpenAI embeddings guide

---

### Week 6 — RAG Architecture

**Learn:**
- RAG pipeline: ingest → chunk → embed → retrieve → generate
- Chunking strategy
- Retrieval quality
- Grounded answer generation
- **Persistence**: store embeddings in a vector DB backed by a file or simple Postgres (pgvector) instead of in-memory only

**Deliverables:**
- RAG module for your job-search system
- Grounded explanations based on retrieved resume/project evidence
- Comparison: LLM-only vs RAG-assisted output
- Embeddings persisted to disk or DB between runs

**Resources:**
- LlamaIndex docs
- OpenAI cookbook — RAG
- pgvector README (optional but useful)

> **Added gap:** Real RAG systems persist their vector index. Using in-memory FAISS only is fine for prototypes but adds one line on your resume if you back it with a database.

---

### Week 7 — RAG Project: Portfolio Document QA

**Learn:**
- Building a document QA app
- Retrieving from personal project documents
- Answering with citations or evidence snippets

**Deliverables:**
- Portfolio QA app over resume, READMEs, and project notes
- Answers include retrieved evidence
- Demo script for interviews

**Resources:**
- LlamaIndex examples
- OpenAI cookbook

---

### Week 8 — Evaluation and Debugging for Generative AI

**Learn:**
- How to evaluate LLM outputs
- Manual and automated eval sets
- Consistency checks
- Debugging prompts and retrieval failures
- **Model selection / cost tradeoffs**: when to use GPT-4o vs GPT-4o-mini vs Claude Sonnet vs Claude Haiku

**Deliverables:**
- Evaluation dataset with 20–30 sample job cases
- Basic scoring rubric
- Before/after accuracy/consistency notes
- Written note: cost-per-run estimate for each model tier you tested

**Resources:**
- Evaluating and Debugging Generative AI — DeepLearning.AI
- Weights & Biases docs

> **Added gap:** Model selection is a real engineering decision. Document the tradeoff you made — this is a talking point interviewers love.

---

## Month 3 — Agents, Tool Use, and Workflow Automation

> Goal: Turn LLM calls into tool-using workflows that can take actions, route decisions, and coordinate multiple steps.

### Week 9 — Function Calling, Tools, and Agents

**Learn:**
- Function calling / tool use concepts
- Tool schemas
- Routing user requests to tools
- Agent vs. workflow distinction

**Deliverables:**
- Tool definitions for job lookup, resume lookup, and fit evaluation
- Agent workflow prototype
- Tool call logs

**Resources:**
- Functions, Tools and Agents with LangChain — DeepLearning.AI
- Claude tool use docs
- OpenAI function calling guide

---

### Week 10 — Agent Workflow Design

**Learn:**
- Planning vs. deterministic workflows
- When NOT to use agents
- Guardrails and allowed actions
- Breaking a task into reliable steps

**Deliverables:**
- Job decision agent v1
- Allowed actions and failure states documented
- Architecture diagram in README

**Resources:**
- Anthropic — writing tools for agents
- OpenAI cookbook — multi-tool orchestration

---

### Week 11 — Memory and Multi-turn AI Interaction

**Learn:**
- Conversation state
- Short-term vs. long-term memory
- User preference handling
- Avoiding unsafe or irrelevant memory use
- **Database persistence for memory**: storing conversation history in SQLite or Postgres

**Deliverables:**
- Interactive job review flow
- Session state for user preferences
- Conversation transcript examples
- Conversation history persisted to a local DB

**Resources:**
- LangChain memory concepts
- Claude prompt engineering docs

> **Added gap:** Conversation history needs to survive process restarts in production. Adding a DB layer here is the right time — you already have context from Week 6.

---

### Week 12 — System Integration and Portfolio Demo

**Learn:**
- Combining LLM, RAG, tools, logging, and evals
- Demo readiness and technical storytelling

**Deliverables:**
- Integrated job-search AI system
- Demo video or demo script
- Final architecture README

**Resources:**
- OpenAI Cookbook
- LlamaIndex docs

---

## Month 4 — Production Readiness, Portfolio, and Interview Prep

> Goal: Make your work look like engineering, not a course project.

### Week 13 — Reliability, Observability, and Fallbacks

**Learn:**
- Failure handling
- Provider fallback strategy (OpenAI fails → switch to Anthropic)
- Logging and tracing
- Cost and latency awareness
- **Prompt caching**: reduce cost on repeated context with Anthropic's cache control or OpenAI's prompt caching

**Deliverables:**
- Fallback path for model or parsing failures
- Logging dashboard or structured logs
- Cost estimate per run
- At least one place in your pipeline using prompt caching

**Resources:**
- LangSmith docs
- OpenAI production best practices
- Anthropic prompt caching docs

> **Added gap:** Provider fallback (OpenAI ↔ Anthropic) is a real production pattern and demonstrates multi-provider literacy.

---

### Week 14 — API Layer and Deployment

**Learn:**
- Serving AI features through APIs with FastAPI
- Environment variables and secret management
- Docker containerization
- Simple cloud deployment: **AWS Lambda, Google Cloud Run, or Render**

**Deliverables:**
- FastAPI endpoint for job evaluation
- Dockerfile that runs the app
- Deployed to at least one cloud provider (Cloud Run is the simplest)
- README with setup instructions
- Optional simple UI or CLI demo

**Resources:**
- FastAPI docs
- Docker docs
- Google Cloud Run quickstart (recommended: free tier, simple deploy)
- Render deploy docs

> **Added gap:** Render is fine for demos, but one Cloud Run or AWS Lambda deploy puts a real cloud provider on your resume. Cloud Run is the easiest path — `docker build → gcloud run deploy`.

---

### Week 15 — Portfolio Packaging and Interview Story

**Learn:**
- How to explain an AI application architecture
- Tradeoffs: LLM vs. rules, RAG vs. fine-tuning, agent vs. workflow
- Reliability, evals, failure modes
- Streaming, cost, and model selection decisions

**Deliverables:**
- Final GitHub README for each project
- One-page project brief
- Interview answers for 10 common AI project questions
- AI-focused resume bullets

**Resources:**
- OpenAI guides
- Anthropic docs

---

### Week 16 — Application Sprint and Iteration

**Deliverables:**
- AI-focused resume version live
- Portfolio link set
- Weekly iteration plan based on recruiter/interview feedback
- Applications submitted to AI Application Engineer, LLM Engineer, AI Software Engineer roles

---

## Execution Rules

1. Do not collect more courses. The bottleneck is implementation.
2. Every resource should be used with a project task. No passive watching.
3. At the end of each study session, write a note in `daily_notes/`: what changed, what works, what failed, what comes next.
4. By Week 4, begin applying lightly. By Week 12, apply seriously. By Week 16, focus on interview conversion.

---

## Suggested Weekly Schedule

| Time | Purpose |
|---|---|
| 2 hours | Watch or read the selected resource for the week |
| 4–6 hours | Implement the weekly project deliverable |
| 1 hour | Write notes, update README, prepare interview explanations |
| 1 hour | Review failures, improve prompts, add validation or tests |

---

## Key Technologies Summary

| Category | Tools |
|---|---|
| LLM APIs | OpenAI API, Anthropic API (Claude) |
| Orchestration | LangChain, LangSmith |
| RAG | LlamaIndex, FAISS, pgvector |
| Embeddings | OpenAI `text-embedding-3-*`, sentence-transformers |
| Agents | LangChain agents, OpenAI function calling, Claude tool use |
| Evaluation | W&B, LangSmith, custom eval sets |
| Backend | FastAPI, Python |
| Database | SQLite, Postgres (for memory/persistence), pgvector |
| Infra | Docker, Google Cloud Run (or AWS Lambda), Render |
| Observability | LangSmith, structured logging |
