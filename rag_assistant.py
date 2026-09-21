import os
import warnings

# Suppress warnings for clean console output
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

GROQ_KEY = "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"
INDEX_PATH = "faiss_index"
DOCS_DIR = "."  # Checks current directory for .txt and .pdf files

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 1. PERSISTENT VECTOR DB LOAD OR BUILD ---
if os.path.exists(INDEX_PATH):
    print("1. Loading cached vector index from disk...")
    db = FAISS.load_local(INDEX_PATH, embedding_function, allow_dangerous_deserialization=True)
    print("? FAISS index loaded in <1s!\n")
else:
    print("1. Loading documents from directory...")
    documents = []
    
    # Load TXT files
    txt_loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    # Load PDF files
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    print(f"2. Splitting {len(documents)} document(s) into vector chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("3. Generating embeddings & building FAISS index...")
    db = FAISS.from_documents(docs, embedding_function)
    
    # Save index to local disk
    db.save_local(INDEX_PATH)
    print("? Vector database index created & saved to disk!\n")

# --- 2. INITIALIZE GROQ LLM ---
print("4. Connecting to Groq Inference Engine...")
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, groq_api_key=GROQ_KEY)

prompt_template = ChatPromptTemplate.from_template(
    """You are an AI Maintenance Assistant. Answer the technician's query strictly based on the provided technical manual context.

Context:
{context}

Question:
{question}

Answer:"""
)

chain = prompt_template | llm

# --- 3. INTERACTIVE CLI LOOP WITH SOURCE CITATIONS ---
print("\n" + "=" * 50)
print("  ??? AI MAINTENANCE ASSISTANT IS READY  ")
print("=" * 50)
print("Type your query below (or 'exit' to quit):\n")

while True:
    try:
        query = input("Technician Query > ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting assistant. Have a safe shift!")
            break

        # Retrieve relevant chunks
        matching_docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in matching_docs])

        # Extract unique sources and pages for attribution
        sources = set()
        for doc in matching_docs:
            source_name = os.path.basename(doc.metadata.get("source", "manual.txt"))
            page_num = doc.metadata.get("page", None)
            if page_num is not None:
                sources.add(f"{source_name} (Page {page_num + 1})")
            else:
                sources.add(source_name)

        # Synthesize answer
        response = chain.invoke({"context": context, "question": query})

        print("\n--- RESPONSE ---")
        print(response.content)
        print("\n?? Sources Referenced:", ", ".join(sources))
        print("-" * 50 + "\n")

    except KeyboardInterrupt:
        print("\nExiting assistant.")
        break
