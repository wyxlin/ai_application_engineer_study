# Embeddings and Vector Search

**Week:** 4
**Status:** Completed

---

## What is an Embedding?

A list of numbers (vector) that represents the **meaning** of text:

```
"Python backend engineer"  → [0.23, -0.81, 0.45, ...]  (1536 numbers)
"Java software developer"  → [0.21, -0.79, 0.41, ...]  (similar → close)
"French cuisine recipe"    → [-0.67, 0.34, -0.22, ...]  (different → far)
```

**Cosine similarity score:** -1 to 1, higher = more similar meaning.

---

## Embedding Model vs LLM

| | LLM (gpt-4o) | Embedding model (text-embedding-3-small) |
|---|---|---|
| Output | Text | Vector (numbers) |
| Purpose | Reasoning, generation | Semantic similarity |
| Speed | Slow | Fast |
| Cost | Expensive | Cheap |
| "Thinks"? | Yes | No — just converts |

---

## Model Selection

| Model | Cost | Quality | Dimensions |
|---|---|---|---|
| `text-embedding-3-small` | Cheap | Good | 1536 |
| `text-embedding-3-large` | 5x more | Better | 3072 |

**Use `text-embedding-3-small`** for job-search assistant — good quality, low cost.

---

## Full Workflow

**Phase 1 — Indexing (once):**
```
500 JDs → embedding model → 500 vectors → stored in FAISS
```

**Phase 2 — Search (every query):**
```
resume → embedding model → 1 vector → FAISS search → top N matches
```

---

## Code

### Setup
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Build index
```python
vector_store = FAISS.from_texts(job_descriptions, embeddings)
```

### Search with score
```python
results = vector_store.similarity_search_with_score(query, k=3)

for result, score in results:
    print(f"Score: {score:.3f} | {result.page_content}")
```

### Threshold filtering
```python
THRESHOLD = 1.2  # L2 distance — lower = more similar

for result, score in results:
    if score < THRESHOLD:
        print(f"Match: {result.page_content}")
    else:
        print("No relevant match found")
```

### Persistence — save and load
```python
# Save after building (embed once)
vector_store.save_local("faiss_index")

# Load on subsequent runs (no API calls)
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

## Key Rules

1. **Vector search always returns results** — never says "no match". Always set a threshold.
2. **FAISS uses L2 distance** — lower score = more similar (opposite of cosine similarity score).
3. **Embed once, search many times** — save index to disk to avoid re-embedding on every run.
4. **Two-layer architecture**: embedding search (fast, cheap) narrows candidates → LLM evaluation (slow, expensive) ranks them.

---

## How it fits in the job-search assistant

```
500 JDs (from Greenhouse/Lever)
    ↓
Embedding model → FAISS index (saved to disk)
    ↓
User resume → embed → search FAISS → top 10 JDs
    ↓
LLM evaluation pipeline → top 3 with fit scores
    ↓
Natural language output to user
```
