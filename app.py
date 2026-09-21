import os
import tempfile
import io
import base64
import html
import csv
import datetime
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


# Custom robust Ensemble Retriever class to eliminate import/version conflicts
class SimpleEnsembleRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        combined_docs = []
        seen_contents = set()
        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    combined_docs.append(doc)
        return combined_docs


# 1. Page Configuration
st.set_page_config(
    page_title="AI Maintenance Assistant",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 AI Maintenance Assistant")
st.caption("Multimodal Field Tool: QR Equipment Lock, Hybrid RAG, Citation Tracking, Maintenance Scheduler, Multi-Language Voice, Vision Analysis, PDF Work Logs, and Technician Feedback.")

# 2. Secure API Key Access
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("`GROQ_API_KEY` not found! Please configure it in Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()


# 3. Helper Function: CSV Feedback Logger & Reader
FEEDBACK_FILE = "feedback_log.csv"

def log_feedback_to_csv(timestamp, user_prompt, assistant_response, rating, equipment_tag, citations):
    file_exists = os.path.exists(FEEDBACK_FILE)
    citation_text = " | ".join([c.page_content[:100].replace("\n", " ") for c in citations]) if citations else "None"
    
    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Equipment Tag", "User Prompt", "Assistant Response", "Rating", "Citations Sample"])
        writer.writerow([timestamp, equipment_tag or "N/A", user_prompt, assistant_response, rating, citation_text])


# 4. Helper Function: PDF Report Generator
def generate_pdf_report(messages, active_tag=None):
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
    meta_info = f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if active_tag:
        meta_info += f" | <b>Equipment Tag:</b> {active_tag}"
    story.append(Paragraph(meta_info, meta_style))
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


# 5. Helper Function: QR/Barcode Detection using OpenCV
def scan_qr_code(image_bytes):
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        if data:
            return data.strip()
    except Exception as e:
        st.sidebar.error(f"QR Scan error: {e}")
    return None


# 6. Sidebar Controls
st.sidebar.header("🌐 Language Settings")
LANGUAGE_MAP = {
    "English 🇺🇸": {"code": "en", "name": "English"},
    "Amharic 🇪🇹": {"code": "am", "name": "Amharic"},
    "Spanish 🇪🇸": {"code": "es", "name": "Spanish"},
    "French 🇫🇷": {"code": "fr", "name": "French"}
}

selected_lang_label = st.sidebar.selectbox("Preferred Language / ቋንቋ", list(LANGUAGE_MAP.keys()), index=0)
selected_lang_code = LANGUAGE_MAP[selected_lang_label]["code"]
selected_lang_name = LANGUAGE_MAP[selected_lang_label]["name"]

st.sidebar.divider()
st.sidebar.header("🏷️ Equipment QR Scanner")

if "active_qr_tag" not in st.session_state:
    st.session_state.active_qr_tag = None

enable_camera = st.sidebar.checkbox("📷 Enable QR Camera Scanner", value=False)

camera_photo = None
if enable_camera:
    camera_photo = st.sidebar.camera_input("Scan Equipment Tag QR")

if camera_photo is not None:
    detected_qr = scan_qr_code(camera_photo.getvalue())
    if detected_qr:
        st.session_state.active_qr_tag = detected_qr
        st.sidebar.success(f"Locked on Tag: **{detected_qr}**")
    else:
        st.sidebar.warning("No QR Code detected in photo. Try adjusting light or focus.")

if st.session_state.active_qr_tag:
    st.sidebar.info(f"🏷️ **Active Target:** `{st.session_state.active_qr_tag}`")
    if st.sidebar.button("Clear QR Target"):
        st.session_state.active_qr_tag = None
        st.rerun()

st.sidebar.divider()
st.sidebar.header("📄 Upload Documentation")
uploaded_file = st.sidebar.file_uploader("Upload manual (PDF or TXT)", type=["pdf", "txt"])

st.sidebar.divider()
st.sidebar.header("⚙️ Equipment Scheduler")
op_hours = st.sidebar.number_input("Current Operating Hours", min_value=0, value=450, step=10)
last_service = st.sidebar.date_input("Last Service Date", value=datetime.date.today() - datetime.timedelta(days=90))

INTERVALS = {
    "Oil & Filter Change": 500,
    "Air Filter Inspection/Replacement": 1000,
    "Hydraulic System Service": 2500,
    "Major Engine Overhaul": 5000
}

gen_checklist = st.sidebar.button("📋 Generate Preventive Checklist")

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

# Manager Feedback Export Control
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        csv_data = f.read()
    st.sidebar.download_button(
        label="📊 Download Feedback Log (CSV)",
        data=csv_data,
        file_name=f"technician_feedback_log_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.sidebar.caption("No technician feedback recorded yet.")


# 7. Helper Function: Multimodal Vision Analysis
def analyze_image_with_groq(image_bytes, user_prompt, lang_name, key, active_tag=None):
    client = Groq(api_key=key)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = user_prompt if user_prompt else f"Inspect this equipment photo. Describe visible defects, rust, wear, cracks, or electrical issues, and list action items."
    if active_tag:
        prompt = f"[Target Equipment Model/Tag: {active_tag}] " + prompt
    if lang_name != "English":
        prompt += f" Respond in {lang_name}."

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


# 8. Helper Function: Speech-to-Text
def transcribe_audio(audio_bytes, lang_code, key):
    client = Groq(api_key=key)
    audio_file = ("audio.wav", audio_bytes, "audio/wav")
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        language=lang_code,
        response_format="text"
    )
    return transcription


# 9. Helper Function: Text-to-Speech
def generate_speech(text, lang_code):
    clean_text = text.replace("#", "").replace("*", "").replace("-", "")
    tts = gTTS(text=clean_text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp


# 10. Persistent Indexing & Hybrid Search Creation
INDEX_DIR = "faiss_index"

@st.cache_resource(show_spinner="Processing documentation for Persistent Hybrid Search...")
def setup_hybrid_retriever(file_bytes=None, file_name=None):
    documents = []
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    elif os.path.exists("manual.txt"):
        loader = TextLoader("manual.txt")
        documents.extend(loader.load())

    if not documents and not os.path.exists(INDEX_DIR):
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents) if documents else []

    if os.path.exists(INDEX_DIR) and not file_bytes:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_DIR)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    if splits:
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3

        ensemble_retriever = SimpleEnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever]
        )
        return ensemble_retriever, "hybrid"
    
    return faiss_retriever, "faiss_only"


# 11. Hybrid Retriever Setup Initialization
ensemble_retriever, search_mode = (
    setup_hybrid_retriever(uploaded_file.getvalue(), uploaded_file.name)
    if uploaded_file is not None
    else setup_hybrid_retriever()
)

if ensemble_retriever:
    st.sidebar.info("⚡ Persistent Hybrid Search (BM25 + FAISS) ready.")
else:
    st.sidebar.warning("Upload a manual or rely on visual analysis.")

# 12. RAG Model Setup
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.1
) if ensemble_retriever else None

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a technical maintenance assistant. Answer the user's question based strictly on the provided context and conversation history.
If you do not know the answer based on the context, state that the information is not available in the manual.

IMPORTANT: You MUST write your entire response in {selected_lang_name}.

Format your response using:
- Markdown headers (###) for main sections or symptoms
- Bullet points (-) for action steps, causes, or requirements
- Bold text (**text**) for part numbers, warnings, or key terms

Context:
{{context}}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
]) if ensemble_retriever else None

# 13. Maintenance Scheduler Display Component
st.subheader("⏱️ Preventive Maintenance Status")

if st.session_state.active_qr_tag:
    st.success(f"🏷️ **Active Machine Target:** `{st.session_state.active_qr_tag}`")

col1, col2, col3, col4 = st.columns(4)

cols = [col1, col2, col3, col4]
idx = 0

overdue_items = []
due_soon_items = []

for task, target_hrs in INTERVALS.items():
    next_due = ((op_hours // target_hrs) + 1) * target_hrs
    hrs_remaining = next_due - op_hours

    with cols[idx]:
        if hrs_remaining <= 0:
            st.metric(label=task, value=f"{next_due} hrs", delta=f"{hrs_remaining} hrs (OVERDUE)", delta_color="inverse")
            overdue_items.append(task)
        elif hrs_remaining <= 50:
            st.metric(label=task, value=f"{next_due} hrs", delta=f"{hrs_remaining} hrs left", delta_color="off")
            due_soon_items.append(task)
        else:
            st.metric(label=task, value=f"{next_due} hrs", delta=f"{hrs_remaining} hrs left")
    idx += 1

if overdue_items:
    st.error(f"⚠️ **Attention Required:** Overdue maintenance detected for: {', '.join(overdue_items)}")
elif due_soon_items:
    st.warning(f"🔔 **Upcoming Maintenance:** Scheduled within 50 operating hours: {', '.join(due_soon_items)}")
else:
    st.success("✅ All scheduled maintenance intervals are within nominal limits.")

st.divider()

# 14. Session State & Chat UI Render
if "messages" not in st.session_state:
    st.session_state.messages = []

# Action triggered by "Generate Preventive Checklist" button
if gen_checklist:
    target_str = f" for equipment [{st.session_state.active_qr_tag}]" if st.session_state.active_qr_tag else ""
    checklist_prompt = f"Generate a comprehensive preventive maintenance checklist in {selected_lang_name}{target_str} currently at {op_hours} operating hours, with last service recorded on {last_service}. Highlight key inspection tasks for overdue or upcoming items."
    st.session_state.messages.append({"role": "user", "content": checklist_prompt})

if st.session_state.messages:
    pdf_data = generate_pdf_report(st.session_state.messages, st.session_state.active_qr_tag)
    st.sidebar.download_button(
        label="📥 Download PDF Work Log",
        data=pdf_data,
        file_name=f"maintenance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.caption("Complete a chat interaction to unlock the PDF report generator.")

# Render previous chat history with Feedback Loop controls
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Reference Sources & Citations"):
                for idx, doc in enumerate(message["sources"], 1):
                    page = doc.metadata.get("page", None)
                    page_str = f" (Page {page + 1})" if page is not None else ""
                    st.markdown(f"**Source {idx}{page_str}:**")
                    st.caption(doc.page_content)
        
        if "audio" in message:
            st.audio(message["audio"], format="audio/mp3")

        # Feedback Loop: Rating buttons for assistant responses
        if message["role"] == "assistant":
            rating_key = f"rating_{msg_idx}"
            user_prev_prompt = st.session_state.messages[msg_idx - 1]["content"] if msg_idx > 0 else "N/A"
            
            if rating_key not in st.session_state:
                st.session_state[rating_key] = None

            f_col1, f_col2, f_col3 = st.columns([1, 1, 10])
            with f_col1:
                if st.button("👍", key=f"up_{msg_idx}"):
                    st.session_state[rating_key] = "thumbs_up"
                    log_feedback_to_csv(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        user_prev_prompt,
                        message["content"],
                        "thumbs_up",
                        st.session_state.active_qr_tag,
                        message.get("sources", [])
                    )
            with f_col2:
                if st.button("👎", key=f"down_{msg_idx}"):
                    st.session_state[rating_key] = "thumbs_down"
                    log_feedback_to_csv(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        user_prev_prompt,
                        message["content"],
                        "thumbs_down",
                        st.session_state.active_qr_tag,
                        message.get("sources", [])
                    )

            if st.session_state[rating_key] == "thumbs_up":
                st.caption("👍 *Feedback recorded: Helpful!*")
            elif st.session_state[rating_key] == "thumbs_down":
                st.caption("👎 *Feedback recorded: Needs improvement.*")

# 15. Input Processing
user_input = st.chat_input(f"Ask a question ({selected_lang_name}) or upload a photo to analyze...")

if audio_record and "bytes" in audio_record:
    with st.spinner(f"Transcribing audio in {selected_lang_name} via Groq Whisper..."):
        try:
            transcribed_text = transcribe_audio(audio_record["bytes"], selected_lang_code, api_key)
            if transcribed_text.strip():
                user_input = transcribed_text.strip()
                st.sidebar.info(f"🎙️ **Transcribed ({selected_lang_name}):** \"{user_input}\"")
        except Exception as e:
            st.sidebar.error(f"Voice transcription error: {e}")

# Process triggered message (from chat input or generate checklist button)
if user_input or uploaded_image or (gen_checklist and st.session_state.messages and st.session_state.messages[-1]["role"] == "user"):
    if gen_checklist and not user_input:
        current_prompt = st.session_state.messages[-1]["content"]
    else:
        current_prompt = user_input if user_input else f"Analyze the attached image for mechanical defects. Provide findings in {selected_lang_name}."
        st.session_state.messages.append({"role": "user", "content": current_prompt})
        with st.chat_message("user"):
            st.markdown(current_prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing request in {selected_lang_name} via Hybrid RAG..."):
            combined_response = ""
            retrieved_docs = []

            # Append QR Equipment Tag filter to user search query if present
            rag_query = f"[{st.session_state.active_qr_tag}] {current_prompt}" if st.session_state.active_qr_tag else current_prompt

            # Visual Defect Analysis
            if uploaded_image:
                st.markdown("### 📸 Visual Defect Inspection")
                vision_analysis = analyze_image_with_groq(
                    uploaded_image.getvalue(), 
                    current_prompt, 
                    selected_lang_name, 
                    api_key, 
                    st.session_state.active_qr_tag
                )
                st.markdown(vision_analysis)
                combined_response += f"### Visual Inspection Findings\n{vision_analysis}\n\n"

            # Manual Documentation Hybrid RAG Search with Citations
            if ensemble_retriever and llm and prompt:
                retrieved_docs = ensemble_retriever.invoke(rag_query)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                chain = prompt | llm | StrOutputParser()
                rag_response = chain.invoke({
                    "context": context_text,
                    "chat_history": chat_history,
                    "question": rag_query
                })
                
                if uploaded_image:
                    st.markdown("### 📄 Related Documentation Guidance")
                st.markdown(rag_response)
                combined_response += f"### Manual Documentation Guidance\n{rag_response}"

                if retrieved_docs:
                    with st.expander("📚 View Reference Sources & Citations"):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            page = doc.metadata.get("page", None)
                            page_str = f" (Page {page + 1})" if page is not None else ""
                            st.markdown(f"**Source {idx}{page_str}:**")
                            st.caption(doc.page_content)
            elif not uploaded_image:
                response_msg = "No active documentation found. Please upload a PDF/TXT manual or an image for inspection."
                st.markdown(response_msg)
                combined_response = response_msg

            # Generate Speech Output in selected language
            audio_fp = generate_speech(combined_response, selected_lang_code)
            st.audio(audio_fp, format="audio/mp3")

            st.session_state.messages.append({
                "role": "assistant",
                "content": combined_response,
                "sources": retrieved_docs,
                "audio": audio_fp
            })
            st.rerun()