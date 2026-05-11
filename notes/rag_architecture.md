# RAG Architecture

**Week:** 5
**Status:** Completed

---

## What is RAG?

> RAG (Retrieval-Augmented Generation) is the process of retrieving relevant data and using it to generate a grounded answer.

We use an embedding model to convert text into vectors, store them in a vector database (like FAISS), and retrieve the most semantically similar content to pass as context to the LLM.

---

## Without RAG vs With RAG

| | Without RAG | With RAG |
|---|---|---|
| LLM knowledge source | Training data only | Retrieved real data |
| Answer quality | Generic | Grounded and specific |
| Works with private data | No | Yes |

---

## The Four Steps

```
1. RETRIEVE  — embed query → FAISS search → top N relevant docs
2. AUGMENT   — inject retrieved docs into prompt as context
3. GENERATE  — LLM reads context → generates grounded answer
4. OUTPUT    — answer references specific retrieved content
```

---

## Code

```python
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Build or load FAISS index
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
    vector_store = FAISS.load_local(str(INDEX_DIR), embeddings,
                                    allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_texts(documents, embeddings)
    vector_store.save_local(str(INDEX_DIR))

# Step 2: Retrieve
retrieved_docs = vector_store.similarity_search(query, k=3)
context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

# Step 3: Augment + Generate
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
    Always ground your answer in the provided context.
    Context: {context}"""),
    ("user", "{query}")
])

chain = prompt | llm
result = chain.invoke({"context": context, "query": query})
print(result.content)
```

---

## How it fits the job-search assistant

```
500 JDs in FAISS
    ↓
User resume → similarity search → top 5 JDs retrieved
    ↓
Top 5 JDs injected into LLM prompt as context
    ↓
LLM evaluates each with fit score + reasoning
    ↓
"Your best match is Stripe (8/10) because..."
```

---

## Key Rules

1. **Always persist the FAISS index** — embed once, load on subsequent runs
2. **Retrieve before generating** — never ask the LLM to work from memory on private data
3. **k value** — start with k=3 to 5; too many retrieved docs overwhelm the context
4. **"Always ground your answer in the provided context"** — add this to your system prompt to prevent hallucination
