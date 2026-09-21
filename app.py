import os
import tempfile
import io
import base64
from PIL import Image
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
st.caption("Multimodal AI Maintenance Assistant: PDF/TXT Manual RAG, Voice Control, Chat Memory, and Visual Defect Analysis.")

# 2. Secure API Key Access
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("`GROQ_API_KEY` not found! Please configure it in Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()

# 3. Sidebar Controls (Document, Audio, and Image Uploads)
st.sidebar.header("📄 Upload Documentation")
uploaded_file = st.sidebar.file_uploader("Upload manual (PDF or TXT)", type=["pdf", "txt"])

st.sidebar.divider()
st.sidebar.header("📸 Visual Defect Inspection")
uploaded_image = st.sidebar.file_uploader("Upload component photo", type=["png", "jpg", "jpeg"])

if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Component Preview", use_container_width=True)

st.sidebar.divider()
st.sidebar.header("🎙️ Hands-Free Voice Control")
audio_record = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="🟩 Stop & Process",
    key="voice_input"
)

# 4. Helper Function: Multimodal Vision Analysis
def analyze_image_with_groq(image_bytes, user_prompt, key):
    client = Groq(api_key=key)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = user_prompt if user_prompt else "Inspect this equipment photo. Describe any visible defects, rust, wear, cracks, or electrical anomalies, and list recommended maintenance steps."
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-instruct",
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content

# 5. Helper Function: Speech to Text (Groq Whisper)
def transcribe_audio(audio_bytes, key):
    client = Groq(api_key=key)
    audio_file = ("audio.wav", audio_bytes, "audio/wav")
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="text"
    )
    return transcription

# 6. Helper Function: Text to Speech (gTTS)
def generate_speech(text):
    clean_text = text.replace("#", "").replace("*", "").replace("-", "")
    tts = gTTS(text=clean_text, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# 7. Helper Function: Index Documents for RAG
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

# 8. Vector Store Initialization
vectorstore = None

if uploaded_file is not None:
    vectorstore = process_file(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Indexed `{uploaded_file.name}` successfully!")
elif os.path.exists("manual.txt"):
    with open("manual.txt", "rb") as f:
        vectorstore = process_file(f.read(), "manual.txt")
    st.sidebar.info("Using default `manual.txt`.")
else:
    st.sidebar.warning("Upload a manual or rely on visual analysis.")

# 9. RAG Chain Setup
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

# 10. Session State & Chat Display
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format="audio/mp3")

# Handle input sources
user_input = st.chat_input("Ask a question or upload a photo to analyze...")

if audio_record and "bytes" in audio_record:
    with st.spinner("Transcribing voice command..."):
        try:
            transcribed_text = transcribe_audio(audio_record["bytes"], api_key)
            if transcribed_text.strip():
                user_input = transcribed_text.strip()
                st.sidebar.info(f"🎙️ **Recorded:** \"{user_input}\"")
        except Exception as e:
            st.sidebar.error(f"Transcription error: {e}")

# 11. Process Execution
if user_input or uploaded_image:
    current_prompt = user_input if user_input else "Analyze the attached image for mechanical defects."
    
    st.session_state.messages.append({"role": "user", "content": current_prompt})
    with st.chat_message("user"):
        st.markdown(current_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing request..."):
            combined_response = ""

            # Vision analysis execution
            if uploaded_image:
                st.markdown("### 📸 Visual Defect Inspection")
                vision_analysis = analyze_image_with_groq(uploaded_image.getvalue(), current_prompt, api_key)
                st.markdown(vision_analysis)
                combined_response += f"### Visual Inspection\n{vision_analysis}\n\n"

            # Manual RAG search execution
            if rag_chain:
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                rag_response = rag_chain.invoke({
                    "question": current_prompt,
                    "chat_history": chat_history
                })
                
                if uploaded_image:
                    st.markdown("### 📄 Related Documentation Findings")
                st.markdown(rag_response)
                combined_response += f"### Documentation Guidance\n{rag_response}"
            elif not uploaded_image:
                response_msg = "No active documentation found. Please upload a PDF/TXT manual or an image for inspection."
                st.markdown(response_msg)
                combined_response = response_msg

            # Audio Speech Output Generation
            audio_fp = generate_speech(combined_response)
            st.audio(audio_fp, format="audio/mp3")

            st.session_state.messages.append({
                "role": "assistant",
                "content": combined_response,
                "audio": audio_fp
            })