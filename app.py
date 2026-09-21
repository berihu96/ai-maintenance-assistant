import os
import tempfile
import io
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Maintenance Assistant",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 RAG Maintenance Assistant")
st.caption("Upload technical manuals (PDF/TXT) or use voice commands for hands-free troubleshooting.")

# 2. Secure API Key Access
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("`GROQ_API_KEY` not found! Please configure it in Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()

# 3. Sidebar File Uploader & Voice Control
st.sidebar.header("📄 Upload Documentation")
uploaded_file = st.sidebar.file_uploader("Upload a manual (PDF or TXT)", type=["pdf", "txt"])

st.sidebar.divider()
st.sidebar.header("🎙️ Hands-Free Voice Control")
audio_record = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="🟩 Stop & Process",
    key="voice_input"
)

# 4. Helper Function: Transcribe Audio via Groq Whisper API
def transcribe_audio(audio_bytes, key):
    client = Groq(api_key=key)
    audio_file = ("audio.wav", audio_bytes, "audio/wav")
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="text"
    )
    return transcription

# 5. Helper Function: Text to Speech (TTS)
def generate_speech(text):
    # Clean text of markdown characters for smoother voice delivery
    clean_text = text.replace("#", "").replace("*", "").replace("-", "")
    tts = gTTS(text=clean_text, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# 6. Helper Function to Build Vector Store
@st.cache_resource(show_spinner="Processing and indexing manual...")
def process_file(file_bytes, file_name):
    documents = []
    file_ext = os.path.splitext(file_name)[1].lower()

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

# 7. Determine Vector Store Source
vectorstore = None

if uploaded_file is not None:
    vectorstore = process_file(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Indexed `{uploaded_file.name}` successfully!")
elif os.path.exists("manual.txt"):
    with open("manual.txt", "rb") as f:
        vectorstore = process_file(f.read(), "manual.txt")
    st.sidebar.info("Using default `manual.txt`.")
else:
    st.sidebar.warning("Please upload a PDF or TXT manual to begin.")

# 8. RAG Chain Initialization with Memory
rag_chain = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical maintenance assistant. Answer the user's question based strictly on the provided context and conversation history.
If you do not know the answer based on the context, state that the information is not available in the manual.

Format your response using:
- Markdown headers (###) for main sections or symptoms
- Bullet points (-) for action steps, causes, or requirements
- Bold text (**text**) for part numbers, warnings, or key terms

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# 9. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format="audio/mp3")

# Handle input from either text chat input or sidebar audio recorder
user_input = st.chat_input("Ask a maintenance or troubleshooting question...")

if audio_record and "bytes" in audio_record:
    with st.spinner("Transcribing audio input..."):
        try:
            transcribed_text = transcribe_audio(audio_record["bytes"], api_key)
            if transcribed_text.strip():
                user_input = transcribed_text.strip()
                st.sidebar.info(f"🎙️ **Recorded:** \"{user_input}\"")
        except Exception as e:
            st.sidebar.error(f"Error transcribing audio: {e}")

# Process query if present
if user_input:
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
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                response = rag_chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history
                })

                # Generate speech audio for assistant response
                audio_fp = generate_speech(response)

                st.markdown(response)
                st.audio(audio_fp, format="audio/mp3")

                # Store content and audio in session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "audio": audio_fp
                })