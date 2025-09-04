import streamlit as st
import openai
import PyPDF2
import io
import json
import os
from fpdf import FPDF
from datetime import datetime
import re
import ast
from streamlit_webrtc import webrtc_streamer
import av

# --- Constants ---
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
DEFAULT_DEMO = """
John Smith
Software Engineer
Experience: 5 years
Skills: Python, Machine Learning, API development, SQL, Cloud Computing
Projects:
- Built a scalable recommendation system for e-commerce.
- Designed deep learning models for image recognition.
Education:
- B.Tech in Computer Science
Achievements:
- Published paper in IEEE AI Conference.
- Lead developer for open source project with 1,000+ stars.
"""
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# --- Utils (same as before, openai new syntax, TTS + Whisper integrated) ---

def get_openai_key():
    key = st.session_state.get("openai_api_key", "")
    if not key:
        key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Please add your OpenAI API key in the sidebar!")
        st.stop()
    return key

def openai_client():
    key = get_openai_key()
    return openai.OpenAI(api_key=key)

def chat_completion(messages, model="gpt-4o", temperature=0.3, max_tokens=1200):
    client = openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
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
        st.warning(f"TTS Error (use text): {e}")
        return None

def transcribe_audio(audio_bytes):
    client = openai_client()
    try:
        with io.BytesIO(audio_bytes) as file:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=file, response_format="text")
            return transcript
    except Exception as e:
        st.warning(f"Whisper transcription failed: {e}")
        return None

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    elif file.name.lower().endswith(".txt"):
        return "\n".join(line.strip() for line in file.read().decode().splitlines() if line.strip())
    else:
        st.error("Unsupported file format. Use PDF or TXT.")
        return None

# Generation, Evaluation, Summary, PDF (use **same** functions as your provided code)
# (Reuse generate_questions, evaluate_answer, summarize_session, generate_pdf, make_session_json from your code.)

# --- UI helpers to organize layout ---

def sidebar():
    st.sidebar.markdown("# AI Interviewer Settings")
    key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Paste your OpenAI API key here")
    st.session_state["openai_api_key"] = key
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Perplexity AI")

def resume_section():
    st.markdown("## Step 1: Upload Resume or Use Demo")
    uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
    use_demo = st.checkbox("Use Demo Resume")
    if uploaded_file:
        resume = extract_text(uploaded_file)
        if resume:
            st.success("Resume uploaded and parsed successfully!")
            st.text_area("Resume Preview", resume, height=200)
            return resume
    elif use_demo:
        st.info("Using demo resume")
        st.text_area("Demo Resume Preview", DEFAULT_DEMO, height=200)
        return DEFAULT_DEMO
    st.warning("Please upload a resume or select demo to proceed.")
    return None

def candidate_section():
    st.markdown("## Step 2: Candidate Information")
    name = st.text_input("Candidate Name", "")
    role = st.text_input("Position / Role", "Software Engineer")
    exp = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager"])
    question_count = st.slider("Number of Interview Questions", 3, 10, 5)
    model_key = st.selectbox("Choose OpenAI Model", list(MODELS.keys()), index=0)
    if not name.strip():
        st.warning("Please enter the candidate's name to continue.")
    return name.strip(), role.strip(), exp, question_count, MODELS[model_key]

def interview_section(questions, resume, model):
    st.markdown("## Step 3: Interview")
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
        st.session_state["answers"] = []
    idx = st.session_state["current_index"]
    if idx >= len(questions):
        st.success("Interview completed! Please check summary below.")
        return st.session_state["answers"]

    q = questions[idx]
    st.subheader(f"Question {idx+1} ({q.get('topic','')} | {q.get('difficulty','')})")
    st.write(q["text"])

    # Play TTS audio
    if f"tts_{idx}" not in st.session_state:
        audio_resp = text_to_speech(q["text"])
        if audio_resp:
            st.session_state[f"tts_{idx}"] = audio_resp
    if st.session_state.get(f"tts_{idx}"):
        st.audio(st.session_state[f"tts_{idx}"].content, format="audio/mp3")

    # Record audio or type answer
    st.markdown("### Record your answer:")

    audio_state = webrtc_streamer(
        key=f"audio_recorder_{idx}",
        mode="audio",
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if audio_state.audio_frames:
        audio_data = b"".join(frame.to_ndarray(format="flt").tobytes() for frame in audio_state.audio_frames)
        st.session_state["audio_data"] = audio_data
        st.success(f"Recorded {len(audio_state.audio_frames)} audio frames")
    else:
        st.session_state["audio_data"] = None

    candidate_answer = None
    if st.session_state.get("audio_data"):
        st.info("Transcribing your answer...")
        transcription = transcribe_audio(st.session_state["audio_data"])
        if transcription:
            candidate_answer = transcription
            st.text_area("Transcribed Answer (editable)", value=candidate_answer, key=f"transcription_{idx}", height=150)
        else:
            candidate_answer = st.text_area("Could not transcribe audio — please type your answer:", key=f"typed_answer_{idx}", height=150)
    else:
        candidate_answer = st.text_area("Or type your answer here:", key=f"typed_answer_{idx}", height=150)

    if st.button("Submit Answer", key=f"submit_{idx}"):
        answer_text = candidate_answer.strip()
        if not answer_text:
            st.warning("Please provide an answer before submitting.")
            st.stop()
        with st.spinner("Evaluating answer..."):
            evaluation = evaluate_answer(q, answer_text, resume, model)
            evaluation["answer"] = answer_text
        st.session_state["answers"].append(evaluation)
        st.session_state["current_index"] += 1
        st.session_state["audio_data"] = None
        st.experimental_rerun()

    st.info("Press 'Submit Answer' when ready to move to next question.")

def summary_section(name, role, exp, resume, questions, answers):
    st.markdown("## Step 4: Summary & Export")
    summary = summarize_session(questions, answers, resume, st.session_state.get("model", "gpt-4o"))
    st.subheader(f"Overall Score: {summary.get('overall_score', '-')}/10")

    st.markdown("**Strengths:**")
    for strength in summary.get("strengths", []):
        st.write(f"- {strength}")

    st.markdown("**Weaknesses:**")
    for weakness in summary.get("weaknesses", []):
        st.write(f"- {weakness}")

    st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")

    st.markdown("**Next Steps:**")
    for step in summary.get("next_steps", []):
        st.write(f"- {step}")

    pdf_buffer = generate_pdf(name, role, exp, questions, answers, summary)
    st.download_button(
        "Download PDF Report",
        pdf_buffer,
        file_name=f"{name.replace(' ','_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

    session_json = make_session_json(name, role, exp, resume, questions, answers)
    json_bytes = json.dumps(session_json, indent=2).encode()
    st.download_button(
        "Download Session JSON",
        json_bytes,
        file_name=f"{name.replace(' ','_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

    if st.button("Save Session"):
        os.makedirs(SESSION_DIR, exist_ok=True)
        filename = os.path.join(SESSION_DIR, f"{name.replace(' ','_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, "w") as f:
            json.dump(session_json, f, indent=2)
        st.success("Session saved! Please reload app to start new interview.")
        # Reset states
        keys_to_clear = ["current_index", "answers", "questions", "resume", "candidate_name", "role", "experience"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.stop()

def main():
    st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")
    sidebar()

    # Resume upload / demo
    if "resume" not in st.session_state:
        resume = resume_section()
        if not resume:
            st.stop()
        st.session_state["resume"] = resume

    # Candidate info
    if "candidate_name" not in st.session_state:
        name, role, exp, qs_count, model = candidate_section()
        if not name:
            st.warning("Please enter candidate name.")
            st.stop()
        st.session_state["candidate_name"] = name
        st.session_state["role"] = role
        st.session_state["experience"] = exp
        st.session_state["num_questions"] = qs_count
        st.session_state["model"] = model

    # Generate questions
    if "questions" not in st.session_state:
        with st.spinner("Generating interview questions..."):
            questions = generate_questions(st.session_state["resume"], st.session_state["role"], st.session_state["experience"], st.session_state["num_questions"], st.session_state["model"])
        if not questions:
            st.error("Failed to generate questions, please check API key and inputs.")
            st.stop()
        st.session_state["questions"] = questions

    # Conduct interview
    if "answers" not in st.session_state:
        st.session_state["answers"] = []

    answers = interview_ui_voice(st.session_state["questions"], st.session_state["resume"], st.session_state["model"])

    # Summary if done
    if answers and len(answers) == len(st.session_state["questions"]):
        summary_section(st.session_state["candidate_name"], st.session_state["role"], st.session_state["experience"], st.session_state["resume"], st.session_state["questions"], answers)

if __name__ == "__main__":
    main()
