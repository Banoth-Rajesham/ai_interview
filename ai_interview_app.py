import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
from datetime import datetime
import base64
import time
from openai import OpenAI

# ======================
# Config & Initialization
# ======================

st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("🤖 AI Interviewer")

client = OpenAI()

# RTC Config (dict, not RTCConfiguration object)
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ======================
# Utility Functions
# ======================

def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )
        return response
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true" controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)


def transcribe_audio(audio_bytes):
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        with open("temp_audio.wav", "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        return None


def evaluate_answer(question, answer, resume_text, model):
    prompt = f"""
    You are an AI interview evaluator.
    Question: {question['text']}
    Candidate's Answer: {answer}
    Resume context: {resume_text}

    Evaluate the answer on:
    1. Correctness
    2. Clarity
    3. Technical depth
    4. Relevance to resume

    Give a score (0-10) and detailed feedback.
    Return JSON with keys: score, feedback.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return eval(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Evaluation Error: {e}")
        return {"score": 0, "feedback": str(e)}

# ======================
# WebRTC Processor
# ======================

class InterviewProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.last_frame_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Throttle snapshots every 5 seconds
        if time.time() - self.last_frame_time > 5:
            st.session_state.proctoring_img = img.copy()
            self.last_frame_time = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def recv_audio(self, frame):
        self.audio_buffer.append(frame.to_ndarray().tobytes())
        return frame

# ======================
# App Sections
# ======================

def interview_section():
    idx = st.session_state.current_q
    questions = st.session_state.get("questions", [])
    if not questions or idx >= len(questions):
        st.session_state.stage = "summary"
        st.rerun()

    q = questions[idx]
    st.header(f"Question {idx+1}/{len(questions)}: {q['topic']} ({q['difficulty']})")
    st.subheader(q['text'])

    # Text-to-speech
    if f"tts_{idx}" not in st.session_state:
        with st.spinner("Generating audio..."):
            audio_response = text_to_speech(q['text'])
            st.session_state[f"tts_{idx}"] = audio_response.content if audio_response else None
    if st.session_state[f"tts_{idx}"]:
        autoplay_audio(st.session_state[f"tts_{idx}"])

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Candidate Live Feed")
        if "audio_buffer" not in st.session_state:
            st.session_state.audio_buffer = []
        if "proctoring_img" not in st.session_state:
            st.session_state.proctoring_img = None

        webrtc_ctx = webrtc_streamer(
            key=f"interview_cam_{idx}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": True},
            processor_factory=lambda: InterviewProcessor(),
            async_processing=False
        )

        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            st.session_state.audio_buffer.extend(webrtc_ctx.video_processor.audio_buffer)
            webrtc_ctx.video_processor.audio_buffer.clear()

    with col2:
        st.markdown("#### Proctoring Snapshot")
        if st.session_state.proctoring_img is not None:
            st.image(st.session_state.proctoring_img,
                     caption=f"Snapshot at {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("Waiting for first candidate snapshot...")

    st.markdown("---")
    if st.button("Stop and Submit Answer", type="primary"):
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            st.session_state.audio_buffer.extend(webrtc_ctx.video_processor.audio_buffer)

        if not st.session_state.audio_buffer:
            st.warning("Please record an answer before submitting.")
            return

        full_audio_bytes = b"".join(st.session_state.audio_buffer)
        st.session_state.audio_buffer = []

        with st.spinner("Transcribing and evaluating your answer..."):
            answer_text = transcribe_audio(full_audio_bytes)
            if answer_text:
                st.info(f"**Transcribed Answer:** {answer_text}")
                evaluation = evaluate_answer(q, answer_text, st.session_state.get('resume', ''), "gpt-4o")
                evaluation["answer"] = answer_text
                st.session_state.answers.append(evaluation)
                st.session_state.current_q += 1
                st.session_state.proctoring_img = None
                st.rerun()
            else:
                st.error("Transcription failed. Please try recording your answer again.")

def summary_section():
    st.header("Interview Summary")
    for i, ans in enumerate(st.session_state.answers):
        st.subheader(f"Question {i+1}")
        st.write(f"Answer: {ans['answer']}")
        st.write(f"Score: {ans['score']}")
        st.write(f"Feedback: {ans['feedback']}")

    avg_score = np.mean([a['score'] for a in st.session_state.answers])
    st.success(f"Average Score: {avg_score:.2f}")

# ======================
# App Logic
# ======================

def app_logic():
    if "stage" not in st.session_state:
        st.session_state.stage = "interview"
        st.session_state.current_q = 0
        st.session_state.answers = []
        st.session_state.questions = [
            {"text": "Tell me about yourself.", "topic": "General", "difficulty": "Easy"},
            {"text": "Explain polymorphism in OOP.", "topic": "Programming", "difficulty": "Medium"},
            {"text": "What is the time complexity of quicksort?", "topic": "Algorithms", "difficulty": "Hard"},
        ]

    if st.session_state.stage == "interview":
        interview_section()
    elif st.session_state.stage == "summary":
        summary_section()

app_logic()
