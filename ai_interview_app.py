# ==============================================================
# 🧠 AI Interviewer with Live Video/Audio — Stable Final Build
# ==============================================================

# 1) Imports
import streamlit as st
import openai
import PyPDF2
import io
import json
import re
import time
from datetime import datetime
from fpdf import FPDF
import base64
import numpy as np
import wave
import av
from typing import Any, Optional, Tuple
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# 2) Page config and constants
st.set_page_config(page_title="🧠 AI Interviewer with Live Video/Audio", layout="wide", page_icon="🧠")
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 3) Utility: stable JSON extraction (first JSON array/object in text)
def extract_first_json_block(text: str) -> Optional[Any]:
    if not text:
        return None
    # Try to find the first {...} or [...] block using a stack-based scan to avoid brittle regex
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        stack = []
        start_idx = None
        for i, ch in enumerate(text):
            if ch == open_char:
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch == close_char and stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # continue searching for a later valid block
                        pass
    return None

# 4) Video processor for periodic snapshots (proctoring)
class InterviewProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.last_snapshot_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            if time.time() - self.last_snapshot_time > 10:
                st.session_state.proctoring_img = frame.to_image()
                self.last_snapshot_time = time.time()
        except Exception:
            pass
        return frame

# 5) OpenAI helpers
def get_openai_key() -> str:
    key = st.session_state.get("openai_api_key") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return key

def openai_client():
    # This uses the modern OpenAI SDK style; adjust if your SDK version differs.
    return openai.OpenAI(api_key=get_openai_key())

def chat_completion(messages, model="gpt-4o", temperature=0.3, max_tokens=1500) -> str:
    client = openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Defensive extraction: handle different SDK variations
        if hasattr(resp, "choices") and resp.choices:
            choice = resp.choices
            content = getattr(getattr(choice, "message", None), "content", None)
            if not content:
                content = getattr(choice, "text", None)
            if not content:
                content = str(resp)
            return content
        return str(resp)
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

def text_to_speech(text: str, voice: str = "alloy") -> Optional[bytes]:
    client = openai_client()
    try:
        res = client.audio.speech.create(model="tts-1", voice=voice, input=text)
        # Some SDKs return a stream-like object with .read(), others a field or base64
        # Try common access patterns safely
        if hasattr(res, "content") and isinstance(res.content, (bytes, bytearray)):
            return bytes(res.content)
        if hasattr(res, "read"):
            return res.read()
        if isinstance(res, dict):
            # If API returns base64 under e.g. "audio"
            audio_b64 = res.get("audio") or res.get("data") or res.get("content")
            if isinstance(audio_b64, str):
                try:
                    return base64.b64decode(audio_b64)
                except Exception:
                    pass
        # Fallback: try to stringify and fail silently
        return None
    except Exception as e:
        st.warning(f"TTS Error: {e}")
        return None

def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    client = openai_client()
    try:
        with io.BytesIO(audio_bytes) as fileobj:
            fileobj.name = "answer.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=fileobj,
                response_format="text",
            )
            # Handle variants
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
def extract_text(file) -> Optional[str]:
    try:
        if getattr(file, "type", "").lower() == "application/pdf" or str(file.name).lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            chunks = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    chunks.append(page_text)
            return "\n".join(chunks).strip()
        elif getattr(file, "type", "").lower() == "text/plain" or str(file.name).lower().endswith(".txt"):
            return file.getvalue().decode("utf-8").strip()
        else:
            st.error("Unsupported file type. Please upload PDF or TXT.")
            return None
    except Exception as e:
        st.error(f"Failed to extract text from resume: {e}")
        return None

# 7) Question generation / evaluation / summary
def generate_questions(resume: str, role: str, num_questions: int, model: str):
    prompt = (
        f"Generate {num_questions} interview questions for a {role} based on this resume:\n{resume}\n"
        "Return a JSON list of questions with fields: text, topic, difficulty."
    )
    messages = [{"role": "user", "content": prompt}]
    content = chat_completion(messages, model=model)
    questions = extract_first_json_block(content)
    if not isinstance(questions, list):
        st.warning("Could not parse questions from AI response.")
        return None
    return questions

def evaluate_answer(question_text: str, answer_text: str, resume: str, model: str):
    prompt = (
        f"Given the resume:\n{resume}\n"
        f"Evaluate the answer to the question:\n{question_text}\n"
        f"Answer:\n{answer_text}\n"
        "Return a JSON with fields: score (1-10), feedback, better_answer."
    )
    messages = [{"role": "user", "content": prompt}]
    content = chat_completion(messages, model=model)
    evaluation = extract_first_json_block(content)
    if not isinstance(evaluation, dict):
        return {"score": 0, "feedback": "No evaluation returned.", "better_answer": ""}
    return evaluation

def summarize_interview(questions, answers, resume: str, model: str):
    transcript_parts = []
    for q, a in zip(questions, answers):
        transcript_parts.append(
            f"Q: {q.get('text','')}\nA: {a.get('answer','')}\nScore: {a.get('score', 0)}/10"
        )
    transcript = "\n\n".join(transcript_parts)
    prompt = (
        f"Summarize the interview based on the resume:\n{resume}\n"
        f"and the transcript:\n{transcript}\n"
        "Return a JSON with overall_score, strengths (list), weaknesses (list), recommendation."
    )
    messages = [{"role": "user", "content": prompt}]
    content = chat_completion(messages, model=model)
    summary = extract_first_json_block(content)
    if not isinstance(summary, dict):
        st.warning("Could not parse summary from AI response.")
        return None
    return summary

# 8) PDF generation
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AI Interview Report", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_pdf(name, role, summary, questions, answers) -> bytes:
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Candidate: {name}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Role: {role}", ln=True)
    pdf.cell(0, 10, f"Overall Score: {summary.get('overall_score', 'N/A')}/10", ln=True)
    pdf.cell(0, 10, f"Recommendation: {summary.get('recommendation', 'N/A')}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detailed Q&A:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        pdf.multi_cell(0, 10, f"Q{i}: {q.get('text','')}")
        pdf.multi_cell(0, 10, f"Answer: {a.get('answer','')}")
        pdf.multi_cell(0, 10, f"Feedback: {a.get('feedback','')} (Score: {a.get('score','N/A')}/10)")
        pdf.ln(5)
    return pdf.output(dest="S").encode("latin-1")

# 9) Convert audio frames -> WAV bytes (best-effort)
def audio_frames_to_wav_bytes(frames) -> Optional[bytes]:
    if not frames:
        return None
    arrays = []
    sample_rate = None
    for f in frames:
        try:
            arr = f.to_ndarray()
        except Exception:
            continue
        # Normalize shape (samples, channels)
        if arr.ndim == 2 and arr.shape <= 2 and arr.shape > arr.shape[1]:
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

# 10) Sidebar
def sidebar():
    st.sidebar.title("Settings")
    key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("openai_api_key", ""),
        help="Key is kept in session only."
    )
    if key:
        st.session_state["openai_api_key"] = key

# 11) Timer display
def display_timer(start_time: float):
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    st.markdown(f"**Interview Duration:** {mins:02d}:{secs:02d}")

# 12) Main app
def main():
    sidebar()
    st.title("🧠 AI Interviewer with Live Video & Audio")

    if "stage" not in st.session_state:
        st.session_state.stage = "setup"

    if st.session_state.stage == "setup":
        st.header("Step 1: Upload Resume and Candidate Info")
        name = st.text_input("Candidate Name", value=st.session_state.get("candidate_name", ""))
        role = st.text_input("Position / Role", value=st.session_state.get("role", "Software Engineer"))
        num_questions = st.slider("Number of Questions", 3, 10, value=5)
        resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])

        if resume_file:
            resume_text = extract_text(resume_file)
            if resume_text:
                st.text_area("Resume Preview", resume_text, height=150)
                if st.button("Generate Questions and Start Interview"):
                    if not name.strip():
                        st.warning("Please enter candidate name.")
                    elif not role.strip():
                        st.warning("Please enter position/role.")
                    else:
                        st.session_state.candidate_name = name.strip()
                        st.session_state.role = role.strip()
                        st.session_state.resume = resume_text
                        st.session_state.num_questions = num_questions
                        with st.spinner("Generating questions..."):
                            questions = generate_questions(resume_text, role, num_questions, MODELS["GPT-4o"])
                        if questions:
                            st.session_state.questions = questions
                            st.session_state.answers = []
                            st.session_state.current_q = 0
                            st.session_state.stage = "interview"
                            st.session_state.interview_start_time = time.time()
                            st.experimental_rerun()
                        else:
                            st.error("Failed to generate questions.")
            else:
                st.warning("Could not extract text from resume.")

    elif st.session_state.stage == "interview":
        questions = st.session_state.get("questions", [])
        answers = st.session_state.get("answers", [])
        current_q = st.session_state.get("current_q", 0)
        display_timer(st.session_state.get("interview_start_time", time.time()))

        if current_q >= len(questions):
            st.session_state.stage = "summary"
            st.experimental_rerun()
            return

        q = questions[current_q]
        st.header(f"Question {current_q + 1} of {len(questions)}")
        st.write(f"**Topic:** {q.get('topic', 'General')}")
        st.write(f"**Difficulty:** {q.get('difficulty', 'Medium')}")
        st.write(f"**Question:** {q.get('text','')}")

        # TTS audio for question
        tts_key = f"tts_{current_q}"
        if tts_key not in st.session_state:
            with st.spinner("Generating question audio..."):
                audio_content = text_to_speech(q.get('text',''))
                st.session_state[tts_key] = audio_content
        if st.session_state.get(tts_key):
            try:
                b64 = base64.b64encode(st.session_state[tts_key]).decode("utf-8")
                st.markdown(
                    f'<audio controls autoplay src="data:audio/mp3;base64,{b64}"></audio>',
                    unsafe_allow_html=True
                )
            except Exception:
                pass

        # Live video + audio capture
        st.markdown("#### Candidate Live Video & Audio (Recording your answer)")
        webrtc_ctx = webrtc_streamer(
            key=f"webrtc_{current_q}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": True},
            processor_factory=InterviewProcessor,
            async_processing=True,
        )

        # Show proctoring snapshot every ~10 seconds
        if st.session_state.get("proctoring_img") is not None:
            st.image(st.session_state.proctoring_img, caption="Proctoring Snapshot (updated every 10s)")

        # Typed answer fallback
        typed_answer = st.text_area("Or type your answer here (optional):", key=f"typed_answer_{current_q}", height=150)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Stop and Submit Answer"):
                wav_bytes = None
                final_answer = None

                # Safely pull audio frames if available
                try:
                    if webrtc_ctx and hasattr(webrtc_ctx, "audio_receiver") and webrtc_ctx.audio_receiver:
                        frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                        wav_bytes = audio_frames_to_wav_bytes(frames) if frames else None
                except Exception:
                    wav_bytes = None

                if wav_bytes:
                    with st.spinner("Transcribing audio..."):
                        final_answer = transcribe_audio(wav_bytes)

                if not final_answer:
                    final_answer = (typed_answer or "").strip() or None

                if not final_answer:
                    st.warning("No audio or typed answer found. Please record or type an answer.")
                else:
                    with st.spinner("Evaluating answer..."):
                        evaluation = evaluate_answer(q.get('text',''), final_answer, st.session_state.get("resume",""), MODELS["GPT-4o"])
                    evaluation["answer"] = final_answer
                    evaluation.setdefault("score", 0)
                    evaluation.setdefault("feedback", "")
                    answers.append(evaluation)
                    st.session_state.answers = answers
                    st.session_state.current_q = current_q + 1
                    st.session_state.proctoring_img = None
                    st.session_state[f"typed_answer_{current_q}"] = ""
                    st.experimental_rerun()

        with col2:
            if st.button("Skip Question"):
                answers.append({"answer": "", "score": 0, "feedback": "Skipped", "better_answer": ""})
                st.session_state.answers = answers
                st.session_state.current_q = current_q + 1
                st.session_state.proctoring_img = None
                st.session_state[f"typed_answer_{current_q}"] = ""
                st.experimental_rerun()

    elif st.session_state.stage == "summary":
        st.header("Interview Summary")
        questions = st.session_state.get("questions", [])
        answers = st.session_state.get("answers", [])
        resume = st.session_state.get("resume", "")

        with st.spinner("Generating summary..."):
            summary = summarize_interview(questions, answers, resume, MODELS["GPT-4o"])

        if summary:
            st.subheader(f"Overall Score: {summary.get('overall_score', 'N/A')}/10")
            st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")
            st.markdown("**Strengths:**")
            for s in summary.get("strengths", []) or []:
                st.write(f"- {s}")
            st.markdown("**Weaknesses:**")
            for w in summary.get("weaknesses", []) or []:
                st.write(f"- {w}")

            pdf_bytes = generate_pdf(
                st.session_state.get("candidate_name", "Candidate"),
                st.session_state.get("role", "Role"),
                summary,
                questions,
                answers,
            )
            st.download_button(
                label="Download Interview Report (PDF)",
                data=pdf_bytes,
                file_name=f"{st.session_state.get('candidate_name', 'candidate')}_interview_report.pdf",
                mime="application/pdf",
            )

            if st.button("Restart Interview"):
                keys_to_clear = [
                    "stage", "candidate_name", "role", "resume", "num_questions",
                    "questions", "answers", "current_q", "interview_start_time",
                    "proctoring_img"
                ]
                for k in keys_to_clear:
                    if k in st.session_state:
                        del st.session_state[k]
                st.experimental_rerun()

# Entry point
if __name__ == "__main__":
    main()
