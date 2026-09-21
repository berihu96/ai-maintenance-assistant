import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
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
st.caption("Upload technical manuals to query troubleshooting procedures and safety guidelines.")

# 2. Secure API Key Access
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("`GROQ_API_KEY` not found! Please configure it in Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()

# 3. Cache Vector Store Creation
@st.cache_resource(show_spinner="Indexing documentation...")
def get_vectorstore():
    documents = []
    
    # Load local manual.txt if present
    if os.path.exists("manual.txt"):
        loader = TextLoader("manual.txt")
        documents.extend(loader.load())
        
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = get_vectorstore()

# 4. RAG Chain Initialization
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

# 5. Session State & Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a maintenance or troubleshooting question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not vectorstore:
        with st.chat_message("assistant"):
            response = "No technical manuals found to query. Please ensure `manual.txt` exists in your repository."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching manual and generating response..."):
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})