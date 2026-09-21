import os
import tempfile
import io
import base64
import html
import datetime
from PIL import Image
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from groq import Groq

# ReportLab for PDF Work Log Export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# 1. Page Configuration
st.set_page_config(
    page_title="AI Maintenance Assistant",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 AI Maintenance Assistant")
st.caption("Multimodal Field Tool: Hybrid RAG, Persistent Vector Storage, Hands-Free Voice Control, Visual Anomaly Detection, and PDF Work Log Generator.")

# 2. Secure API Key Access
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("`GROQ_API_KEY` not found! Please configure it in Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()

# 3. Helper Function: PDF Report Generator
def generate_pdf_report(messages):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A")
    )
    meta_style = ParagraphStyle(
        'ReportMeta',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.gray
    )
    user_style = ParagraphStyle(
        'UserMsg',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=6
    )
    assistant_style = ParagraphStyle(
        'AssistantMsg',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B")
    )

    story.append(Paragraph("🔧 AI Maintenance Assistant - Work Log Report", title_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))

    for msg in messages:
        role_label = "<b>Technician Inquiry:</b>" if msg["role"] == "user" else "<b>Assistant Finding:</b>"
        style = user_style if msg["role"] == "user" else assistant_style
        
        text = html.escape(msg["content"])
        text = text.replace("\n", "<br/>")
        
        story.append(Paragraph(f"{role_label}<br/>{text}", style))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer

# 4. Sidebar Controls
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

st.sidebar.divider()
st.sidebar.header("📋 Export Maintenance Summary")

# 5. Helper Function: Multimodal Vision Analysis (Groq Llama 3.2 Vision)
def analyze_image_with_groq(image_bytes, user_prompt, key):
    client = Groq(api_key=key)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = user_prompt if user_prompt else "Inspect this equipment photo. Describe visible defects, rust, wear, cracks, or electrical issues, and list action items."
    
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

# 6. Helper Function: Speech-to-Text (Groq Whisper)
def transcribe_audio(audio_bytes, key):
    client = Groq(api_key=key)
    audio_file = ("audio.wav", audio_bytes, "audio/wav")
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="text"
    )
    return transcription

# 7. Helper Function: Text-to-Speech (gTTS)
def generate_speech(text):
    clean_text = text.replace("#", "").replace("*", "").replace("-", "")
    tts = gTTS(text=clean_text, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# 8. Persistent Indexing & Hybrid Search Creation
INDEX_DIR = "faiss_index"

@st.cache_resource(show_spinner="Processing documentation for Persistent Hybrid Search...")
def setup_hybrid_retriever(file_bytes=None, file_name=None):
    documents = []
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Custom uploaded file handling
    if file_bytes and file_name:
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
    # Default manual fallback
    elif os.path.exists("manual.txt"):
        loader = TextLoader("manual.txt")
        documents.extend(loader.load())

    if not documents and not os.path.exists(INDEX_DIR):
        return None, None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents) if documents else []

    # Persistence handling: Load existing index or create and save new one
    if os.path.exists(INDEX_DIR) and not file_bytes:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_DIR)

    # Dense FAISS Retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Sparse BM25 Retriever
    if splits:
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3

        # Hybrid Ensemble Retriever (50% Dense, 50% Sparse BM25 Keyword)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever, "hybrid"
    
    return faiss_retriever, "faiss_only"

# 9. Hybrid Retriever Setup Initialization
ensemble_retriever = None
search_mode = None

if uploaded_file is not None:
    ensemble_retriever, search_mode = setup_hybrid_retriever(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Indexed `{uploaded_file.name}` (Hybrid Search active)!")
else:
    ensemble_retriever, search_mode = setup_hybrid_retriever()
    if ensemble_retriever:
        st.sidebar.info("⚡ Persistent Hybrid Search (BM25 + FAISS) ready.")
    else:
        st.sidebar.warning("Upload a manual or rely on visual analysis.")

# 10. RAG Chain Setup
rag_chain = None
if ensemble_retriever:
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
            "context": (lambda x: x["question"]) | ensemble_retriever | format_docs,
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# 11. Session State & Chat UI Render
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar PDF Download Button
if st.session_state.messages:
    pdf_data = generate_pdf_report(st.session_state.messages)
    st.sidebar.download_button(
        label="📥 Download PDF Work Log",
        data=pdf_data,
        file_name=f"maintenance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.caption("Complete a chat interaction to unlock the PDF report generator.")

# Render previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format="audio/mp3")

# 12. Input Processing
user_input = st.chat_input("Ask a question or upload a photo to analyze...")

if audio_record and "bytes" in audio_record:
    with st.spinner("Transcribing audio input via Groq Whisper..."):
        try:
            transcribed_text = transcribe_audio(audio_record["bytes"], api_key)
            if transcribed_text.strip():
                user_input = transcribed_text.strip()
                st.sidebar.info(f"🎙️ **Transcribed:** \"{user_input}\"")
        except Exception as e:
            st.sidebar.error(f"Voice transcription error: {e}")

if user_input or uploaded_image:
    current_prompt = user_input if user_input else "Analyze the attached image for mechanical defects."
    
    st.session_state.messages.append({"role": "user", "content": current_prompt})
    with st.chat_message("user"):
        st.markdown(current_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing request via Hybrid RAG..."):
            combined_response = ""

            # Visual Defect Analysis
            if uploaded_image:
                st.markdown("### 📸 Visual Defect Inspection")
                vision_analysis = analyze_image_with_groq(uploaded_image.getvalue(), current_prompt, api_key)
                st.markdown(vision_analysis)
                combined_response += f"### Visual Inspection Findings\n{vision_analysis}\n\n"

            # Manual Documentation Hybrid RAG Search
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
                    st.markdown("### 📄 Related Documentation Guidance")
                st.markdown(rag_response)
                combined_response += f"### Manual Documentation Guidance\n{rag_response}"
            elif not uploaded_image:
                response_msg = "No active documentation found. Please upload a PDF/TXT manual or an image for inspection."
                st.markdown(response_msg)
                combined_response = response_msg

            # Generate Speech Output
            audio_fp = generate_speech(combined_response)
            st.audio(audio_fp, format="audio/mp3")

            st.session_state.messages.append({
                "role": "assistant",
                "content": combined_response,
                "audio": audio_fp
            })
            st.rerun()