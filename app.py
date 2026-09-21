import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Maintenance Assistant",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 RAG Maintenance Assistant")
st.caption("Upload technical manuals (PDF or TXT) to query troubleshooting procedures and safety guidelines.")

# 2. Secure API Key Access
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("`GROQ_API_KEY` not found! Please configure it in Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()

# 3. Sidebar File Uploader
st.sidebar.header("📄 Upload Documentation")
uploaded_file = st.sidebar.file_uploader("Upload a manual (PDF or TXT)", type=["pdf", "txt"])

# 4. Helper Function to Build Vector Store from Uploaded File or Default File
@st.cache_resource(show_spinner="Processing and indexing manual...")
def process_file(file_bytes, file_name):
    documents = []
    file_ext = os.path.splitext(file_name)[1].lower()

    # Save uploaded file temporarily to process with LangChain loaders
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
            documents.extend(loader.load())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# 5. Determine Vector Store Source
vectorstore = None

if uploaded_file is not None:
    # Process user uploaded file
    vectorstore = process_file(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Indexed `{uploaded_file.name}` successfully!")
elif os.path.exists("manual.txt"):
    # Fallback to local manual.txt if present
    with open("manual.txt", "rb") as f:
        vectorstore = process_file(f.read(), "manual.txt")
    st.sidebar.info("Using default `manual.txt`.")
else:
    st.sidebar.warning("Please upload a PDF or TXT manual to begin.")

# 6. RAG Chain Initialization
rag_chain = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.1
    )

    prompt_template = """
    You are a technical maintenance assistant. Answer the user's question based strictly on the provided context.
    If you do not know the answer based on the context, state that the information is not available in the manual.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# 7. Session State & Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a maintenance or troubleshooting question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not rag_chain:
        with st.chat_message("assistant"):
            response = "No active documentation found. Please upload a PDF or TXT manual using the sidebar."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating response..."):
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})