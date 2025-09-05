# ==============================================================
# 🧠 AI INTERVIEWER APP — Production-hardened final version with Google OAuth login
# ==============================================================

# 1) IMPORTS
import streamlit as st
import openai
import PyPDF2
import io, json, os, re, time
from fpdf import FPDF
from datetime import datetime
import av
import base64
import numpy as np
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from streamlit_oauth import oauth_login  # Added for Google OAuth
import yaml
from yaml.loader import SafeLoader

# 2) CONFIG & CONSTANTS
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 3) Video processor (must subclass VideoProcessorBase and return VideoFrame)
class InterviewProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.last_proctor_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            if time.time() - self.last_proctor_time > 10:
                st.session_state.proctoring_img = frame.to_image()
                self.last_proctor_time = time.time()
        except Exception:
            pass
        return frame

# 4) Load config.yaml for other config or skip if not needed
if os.path.exists("config.yaml"):
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
else:
    config = {}  # Empty if you don't need legacy config

# 5) OpenAI helpers
def get_openai_key():
    key = st.session_state.get("openai_api_key", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        st.error("Please add your OpenAI API key in the sidebar!")
        st.stop()
    return key

def openai_client():
    return openai.OpenAI(api_key=get_openai_key())

def chat_completion(messages, model="gpt-4o", temperature=0.3, max_tokens=1500):
    client = openai_client()
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        st.stop()

def text_to_speech(text, voice="alloy"):
    client = openai_client()
    try:
        res = client.audio.speech.create(model="tts-1", voice=voice, input=text)
        return res
    except Exception as e:
        st.warning(f"TTS Error: {e}")
        return None

def transcribe_audio(audio_bytes):
    client = openai_client()
    try:
        with io.BytesIO(audio_bytes) as file:
            file.name = "interview_answer.wav"
            transcript = client.audio.transcriptions.create(model="whisper-1", file=file, response_format="text")
            if isinstance(transcript, str):
                return transcript
            if hasattr(transcript, "text"):
                return transcript.text
            if isinstance(transcript, dict) and "text" in transcript:
                return transcript["text"]
            return str(transcript)
    except Exception as e:
        st.warning(f"Whisper transcription failed: {e}")
        return None

# 6) Resume extraction
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        text = "".join(page.extract_text() or "" for page in PyPDF2.PdfReader(file).pages)
    elif file.name.lower().endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        return None
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# 7) Question generation/evaluation/summarize (defensive parsing)
def generate_questions(resume, role, experience, num_questions, model):
    prompt = f"Generate {num_questions} interview questions for a {role} ({experience}) based on this resume: {resume}. Return JSON list of {{'text','topic','difficulty'}}."
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.5)
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "") or str(response)
        match = re.search(r'\[.*\]', content, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return None

def evaluate_answer(question, answer, resume, model):
    prompt = f"Evaluate answer. Resume: {resume}\nQuestion: {question['text']}\nAnswer: {answer}\nReturn JSON with 'score'(1-10),'feedback','better_answer'."
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.2)
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "") or str(response)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        return json.loads(match.group(0)) if match else {"score": 0, "feedback": "No structured response", "better_answer": ""}
    except Exception as e:
        return {"score": 0, "feedback": f"Evaluation error: {e}", "better_answer": "N/A"}

def summarize_session(questions, answers, resume, model):
    transcript = "\n".join(f"Q: {q['text']}\nA: {a['answer']}\nScore: {a.get('score',0)}/10" for q, a in zip(questions, answers))
    prompt = f"Summarize the interview. Resume: {resume}. Transcript: {transcript}. Return JSON with 'overall_score','strengths','weaknesses','recommendation'."
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model)
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "") or str(response)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        return json.loads(match.group(0)) if match else {}
    except Exception as e:
        st.error(f"Error summarizing session: {e}")
        return {}

# 8) PDF generator
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Interview Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(name, role, summary, questions, answers):
    pdf = PDF()
    pdf.add_page()
    def write_text(text):
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
    pdf.set_font('Arial', 'B', 16)
    write_text(f"Candidate: {name}")
    pdf.set_font('Arial', '', 12)
    write_text(f"Role: {role}\nOverall Score: {summary.get('overall_score','N/A')}/10\nRecommendation: {summary.get('recommendation','N/A')}\nDate: {datetime.now().strftime('%Y-%m-%d')}")
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    write_text("Detailed Question & Answer Analysis")
    for i, (q, a) in enumerate(zip(questions, answers)):
        pdf.set_font('Arial', 'B', 12)
        write_text(f"Q{i+1}: {q.get('text','')}")
        pdf.set_font('Arial', '', 12)
        write_text(f"Answer: {a.get('answer','')}")
        pdf.set_font('Arial', 'I', 12)
        write_text(f"Feedback: {a.get('feedback','')} (Score: {a.get('score','N/A')}/10)")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# 9) Sidebar UI
def sidebar():
    st.sidebar.markdown(f"Welcome *{st.session_state.get('name','Guest')}*")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interview Settings")
    st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Paste key here")

# 10) Convert audio frames -> WAV bytes
def audio_frames_to_wav_bytes(frames):
    if not frames:
        return None
    arrays = []
    sample_rate = None
    for f in frames:
        try:
            arr = f.to_ndarray()
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        arrays.append(arr)
        sample_rate = getattr(f, "rate", None) or getattr(f, "sample_rate", sample_rate)
    if not arrays:
        return None
    data = np.concatenate(arrays, axis=0)
    if np.issubdtype(data.dtype, np.floating):
        data = (data * 32767).astype(np.int16)
    else:
        data = data.astype(np.int16)
    nch = data.shape[1] if data.ndim == 2 else 1
    sr = int(sample_rate or 48000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()

# 11) APP SECTIONS: setup, interview, summary
def setup_section():
    st.header("Step 1: Resume and Candidate Details")
    st.session_state['name'] = st.text_input("Candidate Name", value=st.session_state.get('name',''))
    role = st.text_input("Position / Role", st.session_state.get('role','Software Engineer'))
    q_count = st.slider("Number of Questions", 3, 10, st.session_state.get('q_count', 5))
    uploaded_file = st.file_uploader("Upload candidate's resume (PDF or TXT)", type=["pdf","txt"])
    if uploaded_file:
        resume_text = extract_text(uploaded_file)
        st.text_area("Resume Preview", resume_text, height=150)
        if st.button("Start Interview"):
            if resume_text and st.session_state.get('name','') and get_openai_key():
                st.session_state.update({
                    "resume": resume_text,
                    "candidate_name": st.session_state.get('name'),
                    "role": role,
                    "q_count": q_count,
                    "answers": [],
                    "current_q": 0,
                    "stage": "interview"
                })
                with st.spinner("Generating personalized questions..."):
                    qs = generate_questions(resume_text, role, "Mid-Level", q_count, MODELS["GPT-4o"])
                    st.session_state.questions = qs or []
                st.experimental_rerun()

def interview_section():
    if "questions" not in st.session_state or not st.session_state.questions:
        st.info("No questions found. Go to Setup to upload resume and generate questions.")
        if st.button("Back to Setup"):
            st.session_state.stage = "setup"
            st.experimental_rerun()
        return

    idx = st.session_state.get("current_q", 0)
    questions = st.session_state.questions
    if idx >= len(questions):
        st.session_state.stage = "summary"
        st.experimental_rerun()
        return

    q = questions[idx]
    st.header(f"Question {idx+1}/{len(questions)}: {q.get('topic','General')} ({q.get('difficulty','Medium')})")
    st.subheader(q.get('text',''))

    # TTS generation
    tts_key = f"tts_{idx}"
    if tts_key not in st.session_state:
        with st.spinner("Generating audio..."):
            audio_resp = text_to_speech(q.get('text',''))
            st.session_state[tts_key] = getattr(audio_resp, "content", None) if audio_resp else None
    if st.session_state.get(tts_key):
        b64 = base64.b64encode(st.session_state[tts_key]).decode("utf-8")
        st.markdown(f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("#### Candidate Live Feed")
        if "proctoring_img" not in st.session_state:
            st.session_state.proctoring_img = None

        webrtc_ctx = webrtc_streamer(
            key=f"interview_cam_{idx}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": True},
            processor_factory=InterviewProcessor,
            async_processing=True,
        )
        if webrtc_ctx and webrtc_ctx.state.playing and hasattr(webrtc_ctx, 'processor') and webrtc_ctx.processor:
            if "audio_buffer" not in st.session_state:
                st.session_state.audio_buffer = []
            st.session_state.audio_buffer.extend(webrtc_ctx.processor.audio_buffer)
            webrtc_ctx.processor.audio_buffer.clear()

    with col2:
        st.markdown("#### Proctoring Snapshot")
        if st.session_state.proctoring_img:
            st.image(st.session_state.proctoring_img, caption=f"Snapshot at {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("Waiting for first candidate snapshot...")

    st.markdown("---")
    answer_text_input = st.text_area("Or type your answer here (optional):", key=f"typed_answer_{idx}", height=150)

    col_submit, col_skip = st.columns(2)
    with col_submit:
        if st.button("Stop and Submit Answer"):
            wav_bytes = None
            try:
                if webrtc_ctx and hasattr(webrtc_ctx, "audio_receiver") and webrtc_ctx.audio_receiver:
                    frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    wav_bytes = audio_frames_to_wav_bytes(frames) if frames else None
            except Exception:
                wav_bytes = None

            final_answer = None
            if wav_bytes:
                with st.spinner("Transcribing audio..."):
                    final_answer = transcribe_audio(wav_bytes)
            if not final_answer:
                final_answer = (answer_text_input or "").strip() or None
            if not final_answer:
                st.warning("No audio or typed answer found. Please record or type an answer.")
            else:
                with st.spinner("Evaluating answer..."):
                    eval_obj = evaluate_answer(q, final_answer, st.session_state.get("resume",""), MODELS["GPT-4o"])
                    eval_obj["answer"] = final_answer
                    eval_obj.setdefault("score", 0)
                    eval_obj.setdefault("feedback", "")
                    st.session_state.answers.append(eval_obj)
                    st.session_state.current_q = st.session_state.get("current_q", 0) + 1
                    st.session_state.proctoring_img = None
                    st.experimental_rerun()

    with col_skip:
        if st.button("Skip Question"):
            st.session_state.current_q = st.session_state.get("current_q", 0) + 1
            st.session_state.proctoring_img = None
            st.experimental_rerun()

def summary_section():
    st.header("Step 3: Interview Summary")
    questions = st.session_state.get("questions", [])
    answers = st.session_state.get("answers", [])
    resume = st.session_state.get("resume","")
    with st.spinner("Generating final summary..."):
        summary = summarize_session(questions, answers, resume, MODELS["GPT-4o"])
    st.subheader(f"Overall Score: {summary.get('overall_score','-')}/10")
    st.markdown(f"**Recommendation:** {summary.get('recommendation','')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths:**")
        for s in summary.get("strengths", []):
            st.write(f"- {s}")
    with col2:
        st.markdown("**Weaknesses:**")
        for w in summary.get("weaknesses", []):
            st.write(f"- {w}")
    pdf_buf = generate_pdf(
        st.session_state.get("candidate_name", st.session_state.get("name","candidate")),
        st.session_state.get("role",""),
        summary, questions, answers
    )
    st.download_button(
        "Download PDF Report", pdf_buf,
        f"{st.session_state.get('candidate_name','candidate')}_Report.pdf",
        mime="application/pdf"
    )
    if st.button("Restart Interview"):
        st.session_state.stage = "setup"
        st.experimental_rerun()

# 12) App orchestrator + auth UI
def app_logic():
    st.title("🧠 AI Interviewer (Final Working Version)")
    if "stage" not in st.session_state:
        st.session_state.stage = "setup"
    if st.session_state.stage == "setup":
        setup_section()
    elif st.session_state.stage == "interview":
        interview_section()
    elif st.session_state.stage == "summary":
        summary_section()

# ---- Google OAuth Login Integration (minimal, replace previous auth) ----
GOOGLE_CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID"  # Put your Google OAuth client ID here
GOOGLE_CLIENT_SECRET = "YOUR_GOOGLE_CLIENT_SECRET"  # Put your Google OAuth client secret here
REDIRECT_URI = "http://localhost:8501/"  # Change to your deployed URL

if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = False

if not st.session_state.authentication_status:
    user_info = oauth_login(
        provider="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=["openid", "email", "profile"],
        open_browser=True,
    )
    if user_info:
        st.session_state.authentication_status = True
        st.session_state.name = user_info.get("name", "")
        st.experimental_rerun()
    else:
        st.info("Please login with Google to continue.")
else:
    sidebar()
    app_logic()
