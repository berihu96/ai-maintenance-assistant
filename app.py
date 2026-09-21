import os
import shutil
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

GROQ_KEY = "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"
INDEX_PATH = "faiss_index"
DOCS_DIR = "."

st.set_page_config(page_title="AI Maintenance Assistant", page_icon="🛠️", layout="wide")

st.title("🛠️ AI Maintenance Assistant")
st.caption("Ask questions strictly grounded in technical manual documentation.")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embedding_function = get_embeddings()

def load_or_build_index():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embedding_function, allow_dangerous_deserialization=True)
    else:
        documents = []
        txt_loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())
        pdf_loader = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embedding_function)
        db.save_local(INDEX_PATH)
        return db

db = load_or_build_index()

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

# --- SIDEBAR WITH FILE UPLOADER & RE-INDEX BUTTON ---
with st.sidebar:
    st.header("📄 Upload New Manuals")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DOCS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s)!")
        st.info("Click 'Force Re-index Documents' below to index the new files.")

    st.divider()
    st.header("⚙️ Controls & Status")
    st.success("Vector DB Index Active")
    
    if st.button("🔄 Force Re-index Documents"):
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
        st.cache_resource.clear()
        st.rerun()

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT ---
if user_query := st.chat_input("Ex: What grease should be used for bearing lubrication?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching manuals..."):
            matching_docs = db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in matching_docs])

            sources = set()
            for doc in matching_docs:
                source_name = os.path.basename(doc.metadata.get("source", "manual.txt"))
                page_num = doc.metadata.get("page", None)
                if page_num is not None:
                    sources.add(f"{source_name} (Page {page_num + 1})")
                else:
                    sources.add(source_name)

            response = chain.invoke({"context": context, "question": user_query})
            
            formatted_response = f"{response.content}\n\n**📍 Sources Referenced:** `{', '.join(sources)}`"
            st.markdown(formatted_response)
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})