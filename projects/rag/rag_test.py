from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Sample JD database
job_descriptions = [
    "Backend Engineer at Stripe: Python, REST APIs, PostgreSQL, distributed systems",
    "Frontend Engineer at Airbnb: React, TypeScript, Next.js, GraphQL",                                                                                                                
    "Data Engineer at Netflix: Python, Spark, Kafka, data pipelines",
    "AI Engineer at OpenAI: Python, LLMs, prompt engineering, backend APIs",                                                                                                           
    "DevOps Engineer at AWS: Kubernetes, Docker, CI/CD, Terraform",  
]

# Step 1: Build or load FAISS index (skip re-embedding when cache exists)
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    vector_store = FAISS.from_texts(job_descriptions, embeddings)
    vector_store.save_local(str(INDEX_DIR))

# Step 2: Retrieve
resume = "Jason Yuan: Python, gRPC, distributed systems, SQL, backend engineering"
retrieved_jds = vector_store.similarity_search(resume, k=3)
context = "\n".join([f"-{doc.page_content}" for doc in retrieved_jds])

# Step 3: Augment + Generate
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([                                                                                                                                            
      ("system", """You are a job match assistant.                                                                                                                                       
      Based on the retrieved job descriptions and the candidate's resume,                                                                                                                
      evaluate each job and recommend the best match.                                                                                                                                    
      Always ground your answer in the provided job descriptions."""),
      ("user", "Retrieved jobs:\n{context}\n\nCandidate resume:\n{resume}")                                                                                                              
  ])  

chain = prompt | llm
result = chain.invoke({"context": context, "resume": resume})
print(result.content)               