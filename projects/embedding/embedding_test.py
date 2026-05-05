"""Minimal embedding + FAISS similarity demo. Requires OPENAI_API_KEY in .env."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

print("embedding_test.py started", flush=True)

# 1. initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 5 simulated job descriptions
job_descriptions = [
    "Backend Engineer at Stripe: Python, REST APIs, SQL, distributed systems.",
    "Frontend Engineer at Airbnb: React, TypeScript, CSS-in-JS, GraphQL, Next.js.",
    "Data Engineer at Google: Python, SQL, BigQuery, AWS, Snowflake.",
    "AI Engineer at OpenAI: Python, OpenAI API, prompt engineering, backend APIs, SQL.",
    "Full Stack Engineer at Netflix: React, Node.js, Python, SQL, Kubernetes.",
]

# 3. Build or load FAISS index (avoid re-embedding when index already on disk)
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"

if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
    print(f"Loading existing index from {INDEX_DIR} (no embedding API calls)...", flush=True)
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    print("Embedding texts and building FAISS index (calls OpenAI API)...", flush=True)
    vector_store = FAISS.from_texts(job_descriptions, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    print(f"Vector store saved to {INDEX_DIR}", flush=True)

# 4. search most similar job description through resume.
resume = "Python, gRPC, distributed systems, SQL, backend engineering"
print("\nSearching for top 3 matching jobs...", flush=True)
results = vector_store.similarity_search_with_score(resume, k=3)

for result, score in results:
    if score < 1.2:
        print(f"Match: (score: {score:.3f}): {result.page_content}")
    else:
        print(f"No match found (score: {score:.3f})")

print("Done.")


