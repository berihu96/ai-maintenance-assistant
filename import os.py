import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

print("1. Loading technical manual...")
loader = TextLoader("manual.txt")
documents = loader.load()

print("2. Splitting text into vector chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

print("3. Generating embeddings & building local FAISS index...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_function)
print("✓ Vector database index created successfully!\n")

query = "High radial vibration (> 2.0 mm/s) detected on drive shaft with elevated bearing temperature"
print(f"=== PROCESSING FAULT QUERY: '{query}' ===")

matching_docs = db.similarity_search(query, k=2)
context = "\n\n".join([doc.page_content for doc in matching_docs])

# 4. Synthesize answer using Groq Cloud API
print("4. Synthesizing answer via Groq API...")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

prompt_template = ChatPromptTemplate.from_template(
    """You are an AI Maintenance Assistant. Answer the technician's query strictly based on the provided technical manual context.

Context:
{context}

Question:
{question}

Answer:"""
)

chain = prompt_template | llm
response = chain.invoke({"context": context, "question": query})

print("\n=== AI MAINTENANCE ASSISTANT RESPONSE ===")
print(response.content)