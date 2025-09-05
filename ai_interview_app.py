# ==============================================================
# 🧠 AI INTERVIEWER APP (Final Version with Step-by-Step Comments)
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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration   # Live video/audio streaming
import streamlit_authenticator as stauth   # Authentication (login/signup)
import yaml                        # For config.yaml file
from yaml.loader import SafeLoader


# ------------------------------
# 2. STREAMLIT PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")

# Models available
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}

# Where to save session files
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# WebRTC (Google STUN server for live video/audio)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ------------------------------
# 3. INTERVIEW PROCESSOR (Handles video/audio frames)
# ------------------------------
class InterviewProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.last_proctor_time = time.time()

    def recv(self, frame):
        if isinstance(frame, av.AudioFrame):
            # Save candidate's microphone audio
            self.audio_buffer.append(frame.to_ndarray().tobytes())
            return frame
        elif isinstance(frame, av.VideoFrame):
            # Take a snapshot every 10 seconds (for proctoring)
            if time.time() - self.last_proctor_time > 10:
                st.session_state.proctoring_img = frame.to_image()
                self.last_proctor_time = time.time()
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
    client = openai_client()
    try:
        with io.BytesIO(audio_bytes) as file:
            file.name = "interview_answer.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=file, response_format="text"
            )
            return transcript
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
# 8. INTERVIEW QUESTION GENERATION
# ------------------------------
def generate_questions(resume, role, experience, num_questions, model):
    prompt = f"""Generate {num_questions} diverse interview questions for a {role} ({experience}) based on this resume: {resume}.
                 Return a JSON list with 'text', 'topic', and 'difficulty'."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.5)
        json_match = re.search(r'\[.*\]', response.choices[0].message.content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else None
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return None

def evaluate_answer(question, answer, resume, model):
    prompt = f"""Evaluate the candidate's answer. Resume: {resume}. Question: {question['text']}. Answer: {answer}.
                 Return JSON with 'score', 'feedback', 'better_answer'."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.2)
        json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception as e:
        return {"score": 0, "feedback": f"Evaluation error: {e}", "better_answer": "N/A"}


# ------------------------------
# 9. SESSION SUMMARY (Final Report)
# ------------------------------
def summarize_session(questions, answers, resume, model):
    transcript = "\n".join(
        f"Q: {q['text']}\nA: {a['answer']}\nScore: {a['score']}/10"
        for q, a in zip(questions, answers)
    )
    prompt = f"""Summarize the interview. Resume: {resume}. Transcript: {transcript}.
                 Return JSON with 'overall_score', 'strengths', 'weaknesses', 'recommendation'."""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model)
        json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
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
        write_text(f"Answer: {a['answer']}")
        pdf.set_font('Arial', 'I', 12)
        write_text(f"Feedback: {a['feedback']} (Score: {a['score']}/10)")
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')


# ------------------------------
# 11. SIDEBAR
# ------------------------------
def sidebar():
    st.sidebar.markdown(f"Welcome *{st.session_state['name']}*")
    authenticator.logout('Logout', 'sidebar', key='logout_button')
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interview Settings")
    st.session_state["openai_api_key"] = st.sidebar.text_input(
        "OpenAI API Key", type="password", placeholder="Paste key here"
    )


# ------------------------------
# 12. MAIN APP LOGIC
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


# ------------------------------
# 13. SECTIONS (Setup → Interview → Summary)
# ------------------------------
def setup_section():
    st.subheader("📄 Setup Interview")
    role = st.text_input("Job Role", "Data Scientist")
    exp = st.selectbox("Experience Level", ["Fresher", "Mid-level", "Senior"])
    resume = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])
    if st.button("Start Interview") and resume:
        st.session_state.resume_text = extract_text(resume)
        st.session_state.role = role
        st.session_state.exp = exp
        st.session_state.stage = "interview"
        st.experimental_rerun()


def interview_section():
    st.subheader("🎥 Live Interview")
    st.write("Your camera & microphone will start below:")

    webrtc_streamer(
        key="interview_cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": True},
        video_processor_factory=InterviewProcessor,
        async_processing=True,
    )

    if st.button("Finish Interview"):
        st.session_state.stage = "summary"
        st.experimental_rerun()


def summary_section():
    st.subheader("📊 Interview Summary")
    st.success("This is where the AI-generated report will appear.")

    if st.button("Restart"):
        st.session_state.stage = "setup"
        st.experimental_rerun()



# ------------------------------
# 14. AUTH LOGIN + RUN APP
# ------------------------------
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

if not st.session_state["authentication_status"]:
    # Login/Register tabs
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        authenticator.login()
        if st.session_state["authentication_status"]:
            st.experimental_rerun()
        elif st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
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
    sidebar()
    app_logic()
