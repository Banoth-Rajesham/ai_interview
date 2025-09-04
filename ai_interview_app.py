import streamlit as st
import openai
import PyPDF2
import io
import json
import os
from fpdf import FPDF
from datetime import datetime
import re
import av
import base64
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import time

# --- Page Config ---
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")

# --- Constants ---
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Top-Level Class Definition for WebRTC ---
class InterviewProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.last_proctor_time = time.time()

    def recv(self, frame):
        if isinstance(frame, av.AudioFrame):
            self.audio_buffer.append(frame.to_ndarray().tobytes())
            return frame
        elif isinstance(frame, av.VideoFrame):
            if time.time() - self.last_proctor_time > 10:
                st.session_state.proctoring_img = frame.to_image()
                self.last_proctor_time = time.time()
            return frame

# --- User Authentication ---
if not os.path.exists('config.yaml'):
    st.error("Fatal Error: `config.yaml` not found. Please create the configuration file.")
    st.stop()

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Utility Functions ---
def get_openai_key():
    key = st.session_state.get("openai_api_key", "")
    if not key and "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    if not key:
        st.error("Please add your OpenAI API key in the sidebar!")
        st.stop()
    return key

def openai_client():
    key = get_openai_key()
    return openai.OpenAI(api_key=key)

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
            return transcript
    except Exception as e:
        st.warning(f"Whisper transcription failed: {e}")
        return None

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        text = "".join(page.extract_text() or "" for page in PyPDF2.PdfReader(file).pages)
    elif file.name.lower().endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        return None
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    md = f"""<audio controls autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></source></audio>"""
    st.markdown(md, unsafe_allow_html=True)

# --- Core AI Logic ---
def generate_questions(resume, role, experience, num_questions, model):
    prompt = f"""Generate {num_questions} diverse interview questions for a {role} ({experience}) based on this resume: {resume}. Return a JSON list of objects, each with 'text', 'topic', and 'difficulty' keys."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.5)
        json_match = re.search(r'\[.*\]', response.choices[0].message.content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else None
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return None

def evaluate_answer(question, answer, resume, model):
    prompt = f"""Evaluate the candidate's answer based on their resume. Resume: {resume}. Question: {question['text']}. Answer: {answer}. Return a JSON object with 'score' (1-10), 'feedback', and a 'better_answer'."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.2)
        json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception as e:
        return {"score": 0, "feedback": f"Evaluation error: {e}", "better_answer": "N/A"}

def summarize_session(questions, answers, resume, model):
    transcript = "\n".join(f"Q: {q['text']}\nA: {a['answer']}\nScore: {a['score']}/10" for q, a in zip(questions, answers))
    prompt = f"""Summarize the interview. Resume: {resume}. Transcript: {transcript}. Return a JSON object with 'overall_score' (1-10), 'strengths' (list of strings), 'weaknesses' (list of strings), and 'recommendation' ('Strong Hire', 'Hire', or 'No Hire')."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model)
        json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception as e:
        st.error(f"Error summarizing session: {e}")
        return {}
        
# --- PDF Generation ---
class PDF(FPDF):
    def header(self): self.set_font('Arial', 'B', 12); self.cell(0, 10, 'AI Interview Report', 0, 1, 'C')
    def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(name, role, summary, questions, answers):
    pdf = PDF()
    pdf.add_page()
    def write_text(text): pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.set_font('Arial', 'B', 16); write_text(f"Candidate: {name}")
    pdf.set_font('Arial', '', 12); write_text(f"Role: {role}\nOverall Score: {summary.get('overall_score', 'N/A')}/10\nRecommendation: {summary.get('recommendation', 'N/A')}\nDate: {datetime.now().strftime('%Y-%m-%d')}")
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14); write_text("Detailed Question & Answer Analysis")
    for i, (q, a) in enumerate(zip(questions, answers)):
        pdf.set_font('Arial', 'B', 12); write_text(f"Q{i+1}: {q['text']}")
        pdf.set_font('Arial', '', 12); write_text(f"Answer: {a['answer']}")
        pdf.set_font('Arial', 'I', 12); write_text(f"Feedback: {a['feedback']} (Score: {a['score']}/10)"); pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# --- Main Application UI ---
def sidebar():
    st.sidebar.markdown(f"Welcome *{st.session_state['name']}*")
    authenticator.logout('Logout', 'sidebar', key='logout_button')
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interview Settings")
    st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Paste key here")

def app_logic():
    st.title("🧠 AI Interviewer")
    if "stage" not in st.session_state: st.session_state.stage = "setup"
    if st.session_state.stage == "setup": setup_section()
    elif st.session_state.stage == "interview": interview_section()
    elif st.session_state.stage == "summary": summary_section()

def setup_section():
    st.header("Step 1: Resume and Candidate Details")
    name = st.text_input("Candidate Name", value=st.session_state.get('name', ''))
    role = st.text_input("Position / Role", "Software Engineer")
    q_count = st.slider("Number of Questions", 3, 10, 5)
    
    uploaded_file = st.file_uploader("Upload candidate's resume (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file:
        resume = extract_text(uploaded_file)
        st.text_area("Resume Preview", resume, height=150)
        
        if st.button("Start Interview", type="primary"):
            if resume and name and get_openai_key():
                st.session_state.update({"resume": resume, "candidate_name": name, "role": role, "q_count": q_count, "answers": [], "current_q": 0, "stage": "interview"})
                with st.spinner("Generating personalized questions..."):
                    st.session_state.questions = generate_questions(resume, role, "Mid-Level", q_count, MODELS["GPT-4o"])
                st.rerun()
            else:
                st.warning("Please ensure all fields are complete and an API key is provided.")

def interview_section():
    idx = st.session_state.current_q
    questions = st.session_state.get("questions", [])
    if not questions or idx >= len(questions): st.session_state.stage = "summary"; st.rerun()

    q = questions[idx]
    st.header(f"Question {idx+1}/{len(questions)}: {q['topic']} ({q['difficulty']})"); st.subheader(q['text'])

    if f"tts_{idx}" not in st.session_state:
        with st.spinner("Generating audio..."):
            audio_response = text_to_speech(q['text'])
            st.session_state[f"tts_{idx}"] = audio_response.content if audio_response else None
    if st.session_state[f"tts_{idx}"]: autoplay_audio(st.session_state[f"tts_{idx}"])

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Candidate Live Feed")
        if "audio_buffer" not in st.session_state: st.session_state.audio_buffer = []
        if "proctoring_img" not in st.session_state: st.session_state.proctoring_img = None
        
        webrtc_ctx = webrtc_streamer(key=f"interview_cam_{idx}", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": True}, processor_factory=InterviewProcessor, async_processing=True)
        
        if webrtc_ctx.state.playing and webrtc_ctx.processor:
            st.session_state.audio_buffer.extend(webrtc_ctx.processor.audio_buffer)
            webrtc_ctx.processor.audio_buffer.clear()

    with col2:
        st.markdown("#### Proctoring Snapshot")
        if st.session_state.proctoring_img: st.image(st.session_state.proctoring_img, caption=f"Snapshot at {datetime.now().strftime('%H:%M:%S')}")
        else: st.info("Waiting for first candidate snapshot...")

    st.markdown("---")
    if st.button("Stop and Submit Answer", type="primary"):
        if webrtc_ctx.state.playing and hasattr(webrtc_ctx, 'processor') and webrtc_ctx.processor:
            st.session_state.audio_buffer.extend(webrtc_ctx.processor.audio_buffer)
        
        if not st.session_state.audio_buffer: st.warning("Please record an answer before submitting."); return
        
        full_audio_bytes = b"".join(st.session_state.audio_buffer)
        st.session_state.audio_buffer = []
        
        with st.spinner("Transcribing and evaluating your answer..."):
            answer_text = transcribe_audio(full_audio_bytes)
            if answer_text:
                st.info(f"**Transcribed Answer:** {answer_text}")
                evaluation = evaluate_answer(q, answer_text, st.session_state.get('resume'), MODELS["GPT-4o"])
                evaluation["answer"] = answer_text
                st.session_state.answers.append(evaluation)
                st.session_state.current_q += 1
                st.session_state.proctoring_img = None
                st.rerun()
            else:
                st.error("Transcription failed. Please try recording your answer again.")

def summary_section():
    st.header("Step 3: Interview Summary")
    with st.spinner("Generating final summary..."):
        summary = summarize_session(st.session_state.questions, st.session_state.answers, st.session_state.resume, MODELS["GPT-4o"])
    st.subheader(f"Overall Score: {summary.get('overall_score', '-')}/10"); st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")
    col1, col2 = st.columns(2)
    with col1: st.markdown("**Strengths:**"); [st.write(f"- {s}") for s in summary.get("strengths", [])]
    with col2: st.markdown("**Weaknesses:**"); [st.write(f"- {w}") for w in summary.get("weaknesses", [])]
    pdf_buffer = generate_pdf(st.session_state.candidate_name, st.session_state.role, summary, st.session_state.questions, st.session_state.answers)
    st.download_button("Download PDF Report", pdf_buffer, f"{st.session_state.candidate_name}_Report.pdf", type="primary")
    if st.button("Start New Interview"):
        keys_to_clear = [k for k in st.session_state.keys() if k not in ['authentication_status', 'name', 'username']]
        for key in keys_to_clear: del st.session_state[key]
        st.rerun()

# --- Main App Execution ---
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

if not st.session_state["authentication_status"]:
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        authenticator.login()
        if st.session_state["authentication_status"]: st.rerun()
        elif st.session_state["authentication_status"] is False: st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None: st.warning('Please enter your username and password.')

    with register_tab:
        st.subheader("Create a New Account")
        try:
            if authenticator.register_user(fields={'Form name': 'Create Account', 'Username': 'username', 'Name': 'name', 'Email': 'email', 'Password': 'password'}):
                st.success('User registered successfully! Please go to the Login tab to sign in.')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
else:
    sidebar()
    app_logic()
