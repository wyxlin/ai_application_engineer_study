import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Load existing portfolio index
INDEX_DIR = Path(__file__).resolve().parent / "portfolio_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    str(INDEX_DIR), embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can answer questions about a candidate.
    Ground your answer in the provided context only.
    context: {context}"""),
    ("user", "{question}")
])
qa_chain = qa_prompt | llm

# Grader chain
class GradeResult(BaseModel):
    score: int
    reason: str

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator. Grade the answer from 1-5.
    5 = complete, accurate, well-grounded                                                                                                                                              
    3 = partially correct                                                                                                                                                              
    1 = wrong or missing key information
    Return structured data."""),                                                                                                                                                       
    ("user", "Question: {question}\nAnswer: {answer}\nExpected keywords: {keywords}")                                                                                                  
  ])   
grader_chain = grader_prompt | llm.with_structured_output(GradeResult)

# Eval dataset
eval_cases = [
    {                                                                                                                                                                                  
        "question": "What programming languages does the candidate know?",
        "expected_keywords": ["Python", "Java", "SQL"]
    },                                                                                                                                                                                 
    {
        "question": "What distributed systems experience does the candidate have?",                                                                                                    
        "expected_keywords": ["gRPC", "distributed"]      
    },                                  
    {
        "question": "What is the candidate's GPA?",                                                                                                                                    
        "expected_keywords": ["4.0"]
    },                                                                                                                                                                                 
    {                                                     
        "question": "What AI tools has the candidate used?",
        "expected_keywords": ["Claude", "LLM"]                                                                                                                                         
    },                                      
    {                                                                                                                                                                                  
        "question": "What is the candidate's highest degree?",
        "expected_keywords": ["Computer Science", "Master"]                                                                                                                            
    },                                  
]

# Run eval
results = []
for case in eval_cases:
    # Get answer
    retrieved = vector_store.similarity_search(case["question"], k=3)
    context = "\n".join([doc.page_content for doc in retrieved])
    answer = qa_chain.invoke({
        "context": context, 
        "question": case["question"]
        }).content
    
    # Grade answer
    grade = grader_chain.invoke({
        "question": case["question"], 
        "answer": answer, 
        "keywords": ", ".join(case["expected_keywords"])
        })

    results.append({
        "question": case["question"],
        "answer": answer,
        "score": grade.score,
        "reason": grade.reason
    })

    print(f"Question: {case['question']}")
    print(f"Score: {grade.score}/5 - {grade.reason}")
    print("-" * 50)

# Summary
avg_score = sum(result["score"] for result in results) / len(results)
print(f"\nAverage score: {avg_score: .1f}/5")
print(f"Cases passed (>=4): {sum(1 for r in results if r['score'] >= 4)}/{len(results)}")

# Save results
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to eval_results.json")



