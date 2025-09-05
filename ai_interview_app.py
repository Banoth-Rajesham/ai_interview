# ==============================================================
# 🧠 AI INTERVIEWER APP (Final Version - Complete & Fixed)
# ==============================================================

# ------------------------------
# 1. IMPORTS (All Required Libraries)
# ------------------------------
import streamlit as st              # For Streamlit web app
import openai                      # To connect with OpenAI models
import PyPDF2                      # To read PDF resumes
import io, json, os, re, time       # Python utilities
from fpdf import FPDF              # To generate PDF reports
from datetime import datetime
import av                          # Audio/Video frames (for webcam + mic)
import base64                      # Encode/Decode audio
import numpy as np                 # For audio array handling
import wave                        # For writing WAV bytes
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase  # Live video/audio streaming
import streamlit_authenticator as stauth   # Authentication (login/signup)
import yaml                        # For config.yaml file
from yaml.loader import SafeLoader


# ------------------------------
# 2. STREAMLIT PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")

MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ------------------------------
# 3. INTERVIEW PROCESSOR (Handles video frames)
#    - MUST subclass VideoProcessorBase and return av.VideoFrame
# ------------------------------
class InterviewProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        # we will not store audio here; audio frames are read from webrtc_ctx.audio_receiver
        self.last_proctor_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # This runs in a worker thread — keep light and defensive.
        try:
            if time.time() - self.last_proctor_time > 10:
                # Save a snapshot (PIL.Image) to session_state for display
                st.session_state.proctoring_img = frame.to_image()
                self.last_proctor_time = time.time()
        except Exception:
            # Prevent exceptions from breaking the pipeline
            pass
        return frame


# ------------------------------
# 4. AUTHENTICATION (Login/Register using config.yaml)
# ------------------------------
if not os.path.exists('config.yaml'):
    st.error("Fatal Error: `config.yaml` not found. Please create the configuration file.")
    st.stop()

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'],
    config['cookie']['key'], config['cookie']['expiry_days']
)


# ------------------------------
# 5. OPENAI CLIENT HELPERS
# ------------------------------
def get_openai_key():
    key = st.session_state.get("openai_api_key", "")
    if not key and "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    if not key:
        st.error("Please add your OpenAI API key in the sidebar!")
        st.stop()
    return key

def openai_client():
    return openai.OpenAI(api_key=get_openai_key())


# ------------------------------
# 6. OPENAI FUNCTIONS (Chat, TTS, Transcription)
# ------------------------------
def chat_completion(messages, model="gpt-4o", temperature=0.3, max_tokens=1500):
    client = openai_client()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens
        )
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
    """Expect valid WAV/PCM bytes. Returns string transcript or None"""
    client = openai_client()
    try:
        with io.BytesIO(audio_bytes) as file:
            file.name = "interview_answer.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=file, response_format="text"
            )
            # The API may return a string or an object with text depending on wrapper
            if isinstance(transcript, str):
                return transcript
            elif hasattr(transcript, "text"):
                return transcript.text
            elif isinstance(transcript, dict) and "text" in transcript:
                return transcript["text"]
            else:
                return str(transcript)
    except Exception as e:
        st.warning(f"Whisper transcription failed: {e}")
        return None


# ------------------------------
# 7. RESUME HANDLING
# ------------------------------
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        text = "".join(page.extract_text() or "" for page in PyPDF2.PdfReader(file).pages)
    elif file.name.lower().endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        return None
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


# ------------------------------
# 8. INTERVIEW QUESTION GENERATION & EVALUATION
# ------------------------------
def generate_questions(resume, role, experience, num_questions, model):
    prompt = f"""Generate {num_questions} diverse interview questions for a {role} ({experience}) based on this resume: {resume}.
Return a JSON list of objects, each with 'text', 'topic', and 'difficulty' keys."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.5)
        # wrapper may place message in response.choices[0].message.content
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "") or str(response)
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else None
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return None

def evaluate_answer(question, answer, resume, model):
    prompt = f"""Evaluate the candidate's answer based on their resume.
Resume: {resume}
Question: {question['text']}
Answer: {answer}
Return a JSON object with 'score' (1-10), 'feedback', and 'better_answer'."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.2)
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "") or str(response)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {"score":0,"feedback":"No structured response","better_answer":""}
    except Exception as e:
        return {"score": 0, "feedback": f"Evaluation error: {e}", "better_answer": "N/A"}


# ------------------------------
# 9. SESSION SUMMARY (Final Report)
# ------------------------------
def summarize_session(questions, answers, resume, model):
    transcript = "\n".join(
        f"Q: {q['text']}\nA: {a['answer']}\nScore: {a.get('score',0)}/10"
        for q, a in zip(questions, answers)
    )
    prompt = f"""Summarize the interview. Resume: {resume}. Transcript: {transcript}.
Return a JSON object with 'overall_score' (1-10), 'strengths' (list), 'weaknesses' (list), and 'recommendation' ('Strong Hire'|'Hire'|'No Hire')."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model)
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "") or str(response)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception as e:
        st.error(f"Error summarizing session: {e}")
        return {}


# ------------------------------
# 10. PDF REPORT GENERATOR
# ------------------------------
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
    write_text(
        f"Role: {role}\nOverall Score: {summary.get('overall_score', 'N/A')}/10\n"
        f"Recommendation: {summary.get('recommendation', 'N/A')}\nDate: {datetime.now().strftime('%Y-%m-%d')}"
    )
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    write_text("Detailed Question & Answer Analysis")
    for i, (q, a) in enumerate(zip(questions, answers)):
        pdf.set_font('Arial', 'B', 12)
        write_text(f"Q{i+1}: {q['text']}")
        pdf.set_font('Arial', '', 12)
        write_text(f"Answer: {a.get('answer','')}")
        pdf.set_font('Arial', 'I', 12)
        write_text(f"Feedback: {a.get('feedback','')} (Score: {a.get('score','N/A')}/10)")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')


# ------------------------------
# 11. SIDEBAR
# ------------------------------
def sidebar():
    st.sidebar.markdown(f"Welcome *{st.session_state.get('name','Guest')}*")
    try:
        authenticator.logout('Logout', 'sidebar', key='logout_button')
    except Exception:
        pass
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interview Settings")
    st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Paste key here")


# ------------------------------
# 12. UTIL: Convert list[av.AudioFrame] -> WAV bytes
# ------------------------------
def audio_frames_to_wav_bytes(frames):
    """
    Convert list of av.AudioFrame to 16-bit PCM WAV bytes.
    Best-effort: aligns channels and sample rates.
    """
    if not frames:
        return None
    arrays = []
    sample_rate = None
    for f in frames:
        try:
            arr = f.to_ndarray()
        except Exception:
            continue
        # handle layout: (channels, samples) or (samples, channels)
        if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        arrays.append(arr)
        # try to read sample rate
        sample_rate = getattr(f, "rate", None) or getattr(f, "sample_rate", sample_rate)
    if not arrays:
        return None
    data = np.concatenate(arrays, axis=0)
    # cast to int16 if floats
    if np.issubdtype(data.dtype, np.floating):
        data = (data * 32767).astype(np.int16)
    else:
        data = data.astype(np.int16)
    nch = data.shape[1] if data.ndim == 2 else 1
    sr = int(sample_rate or 48000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


# ------------------------------
# 13. MAIN APP LOGIC & SECTIONS
# ------------------------------
def app_logic():
    st.title("🧠 AI Interviewer (v4 - Final Working Version)")
    if "stage" not in st.session_state:
        st.session_state.stage = "setup"
    if st.session_state.stage == "setup":
        setup_section()
    elif st.session_state.stage == "interview":
        interview_section()
    elif st.session_state.stage == "summary":
        summary_section()


def setup_section():
    st.header("Step 1: Resume and Candidate Details")
    # Candidate fields
    st.session_state['name'] = st.text_input("Candidate Name", value=st.session_state.get('name', ''))
    role = st.text_input("Position / Role", st.session_state.get('role', 'Software Engineer'))
    q_count = st.slider("Number of Questions", 3, 10, st.session_state.get('q_count', 5))
    uploaded_file = st.file_uploader("Upload candidate's resume (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file:
        resume = extract_text(uploaded_file)
        st.text_area("Resume Preview", resume, height=150)
        if st.button("Start Interview"):
            if resume and st.session_state.get('name', '') and get_openai_key():
                st.session_state.update({
                    "resume": resume,
                    "candidate_name": st.session_state.get('name'),
                    "role": role,
                    "q_count": q_count,
                    "answers": [],
                    "current_q": 0,
                    "stage": "interview"
                })
                with st.spinner("Generating personalized questions..."):
                    questions = generate_questions(resume, role, "Mid-Level", q_count, MODELS["GPT-4o"])
                    st.session_state.questions = questions or []
                st.experimental_rerun()


def interview_section():
    # Safety checks
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
    st.subheader(q.get('text', ''))

    # Generate TTS if not present
    tts_key = f"tts_{idx}"
    if tts_key not in st.session_state:
        with st.spinner("Generating audio..."):
            audio_response = text_to_speech(q.get('text', ''))
            st.session_state[tts_key] = getattr(audio_response, "content", None) if audio_response else None

    # Play TTS if available
    if st.session_state.get(tts_key):
        b64 = base64.b64encode(st.session_state[tts_key]).decode("utf-8")
        st.markdown(f"""<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Candidate Live Feed")
        if "proctoring_img" not in st.session_state:
            st.session_state.proctoring_img = None

        webrtc_ctx = webrtc_streamer(
            key=f"interview_cam_{idx}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=InterviewProcessor,
            async_processing=True,
            audio_receiver_size=1024
        )

        # show status
        if webrtc_ctx:
            st.write(f"WebRTC state: {webrtc_ctx.state.name if hasattr(webrtc_ctx,'state') else 'unknown'}")

    with col2:
        st.markdown("#### Proctoring Snapshot")
        if st.session_state.proctoring_img:
            st.image(st.session_state.proctoring_img, caption=f"Snapshot at {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("Waiting for first candidate snapshot...")

    st.markdown("---")
    answer_textarea = st.text_area("Or type your answer here (optional):", key=f"typed_answer_{idx}", height=150)

    col_submit, col_skip = st.columns(2)
    with col_submit:
        if st.button("Stop and Submit Answer"):
            # Try to fetch audio frames from audio_receiver
            wav_bytes = None
            try:
                if webrtc_ctx and hasattr(webrtc_ctx, "audio_receiver") and webrtc_ctx.audio_receiver:
                    frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    wav_bytes = audio_frames_to_wav_bytes(frames) if frames else None
            except Exception as e:
                # fallback: no audio frames
                wav_bytes = None

            # Prefer transcribed audio; otherwise use typed answer
            answer_text = None
            if wav_bytes:
                with st.spinner("Transcribing audio..."):
                    answer_text = transcribe_audio(wav_bytes)
            if not answer_text:
                answer_text = answer_textarea.strip() or None

            if not answer_text:
                st.warning("No audio or typed answer found. Please record or type an answer.")
            else:
                with st.spinner("Evaluating answer..."):
                    evaluation = evaluate_answer(q, answer_text, st.session_state.get("resume", ""), MODELS["GPT-4o"])
                    # ensure keys exist
                    evaluation["answer"] = answer_text
                    evaluation.setdefault("score", 0)
                    evaluation.setdefault("feedback", "")
                    st.session_state.answers.append(evaluation)
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
    resume = st.session_state.get("resume", "")
    with st.spinner("Generating final summary..."):
        summary = summarize_session(questions, answers, resume, MODELS["GPT-4o"])
    st.subheader(f"Overall Score: {summary.get('overall_score', '-')}/10")
    st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths:**")
        for s in summary.get("strengths", []):
            st.write(f"- {s}")
    with col2:
        st.markdown("**Weaknesses:**")
        for w in summary.get("weaknesses", []):
            st.write(f"- {w}")

    pdf_buffer = generate_pdf(st.session_state.get("candidate_name", st.session_state.get("name","candidate")), st.session_state.get("role",""), summary, questions, answers)
    st.download_button("Download PDF Report", pdf_buffer, f"{st.session_state.get('candidate_name','candidate')}_Report.pdf", mime="application/pdf")

    if st.button("Restart Interview"):
        st.session_state.stage = "setup"
        st.experimental_rerun()


# ------------------------------
# 14. AUTH LOGIN + RUN APP
# ------------------------------
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

if not st.session_state["authentication_status"]:
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        try:
            name, authentication_status, username = authenticator.login('Login', 'main')
        except Exception:
            # If wrapper signature differs, call with no return; the library may set st.session_state in-place
            try:
                authenticator.login()
            except Exception:
                pass

        if st.session_state.get("authentication_status"):
            st.experimental_rerun()
        elif st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')
        else:
            st.warning('Please enter your username and password.')

    with register_tab:
        st.subheader("Create a New Account")
        try:
            if authenticator.register_user(fields={
                'Form name': 'Create Account',
                'Username': 'username',
                'Name': 'name',
                'Email': 'email',
                'Password': 'password'
            }):
                st.success('User registered successfully! Please go to the Login tab to sign in.')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

else:
    # Show sidebar and run app logic
    sidebar()
    app_logic()
