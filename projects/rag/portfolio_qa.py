from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent / "portfolio_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 1: Load and chunk documents
if not(INDEX_DIR.exists() and any(INDEX_DIR.iterdir())):
    loader = TextLoader("documents/resume.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    print("Index built and saved")
else:
    vector_store = FAISS.load_local(
        str(INDEX_DIR), embeddings, 
        allow_dangerous_deserialization=True
    )
    print("Index loaded from cache")

# Step 2: Retrieve + Generate
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about a candidate.
    Always ground your answer in the provided context.
    If the context doesn't contain enough information, say so.
    
    Context:
    {context}"""),
    ("user", "{question}")
])

chain = prompt | llm

# Step 3: Ask questions
questions = [
    "What distributed systems experience does this candidate have?",
    "What programming languages does the candidate know?",                                                                                                                             
    "What AI-related experience does the candidate have?",
]

for question in questions:
    retrieved = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in retrieved])
    result = chain.invoke({
        "context": context,
        "question": question
    })
    print(f"\nQuestion: {question}")
    print(f"Answer: {result.content}")
    print("-" * 50)
