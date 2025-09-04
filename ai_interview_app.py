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
import numpy as np
from pydub import AudioSegment

# --- Config and constants ---
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

# --- Utility Functions ---

def get_openai_key():
    key = st.session_state.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        st.error("OpenAI API key missing! Set it in sidebar or streamlit secrets.")
        st.stop()
    return key

def extract_text_from_file(f):
    if f.name.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(f)
        txt = ""
        for p in reader.pages:
            txt += p.extract_text() or ""
        return "\n".join([line.strip() for line in txt.splitlines() if line.strip()])
    elif f.name.lower().endswith(".txt"):
        return "\n".join([line.strip() for line in f.read().decode().splitlines() if line.strip()])
    else:
        st.error("Only PDF or TXT files supported.")
        return None

# OpenAI client singleton for new SDK
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
        st.error(f"OpenAI ChatCompletion error: {e}")
        st.stop()

def text_to_speech(text, voice="alloy"):
    client = openai_client()
    try:
        res = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        return res
    except Exception as e:
        st.warning(f"TTS error (fall back to text): {e}")
        return None

def transcribe_audio_whisper(audio_bytes):
    client = openai_client()
    try:
        with io.BytesIO(audio_bytes) as wav_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_file,
                response_format="text"
            )
            return transcription
    except Exception as e:
        st.warning(f"Whisper transcription failed: {e}")
        return None

# Generate questions function (same structure as before)

def generate_questions(resume, role, exp, count, model):
    prompt = f"""
You are an expert interviewer. Generate {count} interview questions with id, text, topic, difficulty, estimated_time_seconds in JSON from resume & role '{role}' and experience '{exp}'. Resume: {resume}
"""
    messages = [{"role": "system", "content": "You generate interview questions."}, {"role":"user", "content": prompt}]
    resp = chat_completion(messages, model=model, max_tokens=2000)
    raw = resp.choices[0].message.content
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group())
            for i, q in enumerate(questions): q["id"] = i + 1
            return questions
        except:
            try:
                questions = ast.literal_eval(match.group())
                for i, q in enumerate(questions): q["id"] = i + 1
                return questions
            except:
                st.error("Parsing questions JSON failed.")
    st.error("Question generation failed.")
    return []

def evaluate_answer(question, answer, resume, model):
    prompt = f"""
Evaluate answer strictly:
Question: {question['text']}
Answer: {answer}
Resume:
{resume}
Return JSON: score 0-10, justification, improvements [3], model_answer.
"""
    messages = [{"role":"system","content":"Evaluate answers."}, {"role":"user","content":prompt}]
    resp = chat_completion(messages, model=model, max_tokens=700)
    raw = resp.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            eval_json = json.loads(match.group())
            eval_json["score"] = int(eval_json.get("score", 0))
            eval_json["improvements"] = eval_json.get("improvements", [])[:3]
            return eval_json
        except:
            try:
                eval_json = ast.literal_eval(match.group())
                eval_json["score"] = int(eval_json.get("score", 0))
                eval_json["improvements"] = eval_json.get("improvements", [])[:3]
                return eval_json
            except:
                return {"score": 0, "justification": "Parsing failed.", "improvements": [], "model_answer": ""}
    return {"score": 0, "justification": "Evaluation failed.", "improvements": [], "model_answer": ""}

def summarize_session(questions, evals, resume, model):
    items = []
    for i, q in enumerate(questions):
        ev = evals[i] if i < len(evals) else {}
        items.append({"question": q["text"], "score": ev.get("score", 0), "justification": ev.get("justification", ""), "improvements": ev.get("improvements", []), "answer": ev.get("answer",""), "model_answer": ev.get("model_answer", "")})
    prompt = f"""
Summarize interview results:
{json.dumps(items, indent=2)}
Candidate resume:
{resume}
Return JSON: overall_score 0-10, strengths [], weaknesses [], recommendation Yes/No/Maybe, next_steps [].
"""
    messages = [{"role":"system", "content":"Summarize interviews."}, {"role":"user","content":prompt}]
    resp = chat_completion(messages, model=model, max_tokens=600)
    raw = resp.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            summ = json.loads(match.group())
            summ["overall_score"] = int(summ.get("overall_score", 5))
            return summ
        except:
            return {"overall_score": 5, "strengths": [], "weaknesses": [], "recommendation": "Maybe", "next_steps": []}
    return {"overall_score": 5, "strengths": [], "weaknesses": [], "recommendation": "Maybe", "next_steps": []}

# PDF generation (same as previous)

# --- Streamlit UI ---

def audio_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    audio = frame.to_ndarray(format='fltp')
    # You can implement visualization or save audio here if desired
    return frame

def interview_ui_voice(questions, resume, model):
    st.header("Step 3: Interview with Voice")
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = 0
        st.session_state["evaluations"] = []

    idx = st.session_state["current_question"]
    if idx >= len(questions):
        st.success("Interview complete!")
        return st.session_state["evaluations"]

    q = questions[idx]

    # Show question text & TTS playback
    st.subheader(f"Question {idx+1} [{q.get('topic','')} | {q.get('difficulty','')}]")
    st.write(q["text"])

    # Text-to-Speech audio playback
    tts_audio = st.session_state.get(f"tts_{idx}")
    if tts_audio is None:
        audio_resp = text_to_speech(q["text"])
        if audio_resp:
            tts_audio = audio_resp
            st.session_state[f"tts_{idx}"] = tts_audio
    if tts_audio:
        st.audio(tts_audio, format='audio/mp3', start_time=0)

    # Audio recorder widget using streamlit-webrtc
    st.markdown("### Record your answer using your mic below:")
    audio_state = webrtc_streamer(
        key=f"webrtc_{idx}",
        mode="audio",
        in_audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    # Buffer to hold recorded audio bytes
    if "audio_bytes" not in st.session_state:
        st.session_state["audio_bytes"] = None

    if audio_state.audio_frames:
        # Concatenate audio frames and convert to wav bytes
        aud_frames = audio_state.audio_frames
        temp_audio = b"".join([f.to_ndarray().tobytes() for f in aud_frames])
        # Use pydub or other method to convert raw audio frames into wav bytes if needed
        # For simplicity, will assume direct raw bytes usage for Whisper
        st.session_state["audio_bytes"] = temp_audio
        st.success(f"Recorded {len(aud_frames)} audio frames")

    candidate_answer = None
    if st.session_state["audio_bytes"]:
        st.info("Transcribing recorded audio...")
        transcription = transcribe_audio_whisper(st.session_state["audio_bytes"])
        if transcription:
            candidate_answer = transcription
            st.text_area("Transcribed Answer (edit if needed):", value=candidate_answer, key=f"transcription_{idx}", height=150)
        else:
            candidate_answer = st.text_area("Could not transcribe. Please type your answer:", key=f"typed_answer_{idx}", height=150)
    else:
        candidate_answer = st.text_area("Type your answer here:", key=f"typed_answer_{idx}", height=150)

    if st.button("Submit Answer", key=f"submit_{idx}"):
        final_answer = candidate_answer.strip() if candidate_answer else ""
        if not final_answer:
            st.warning("Please provide an answer before submitting.")
            st.stop()
        with st.spinner("Evaluating answer..."):
            evaluation = evaluate_answer(q, final_answer, resume, model)
            evaluation["answer"] = final_answer
        st.session_state["evaluations"].append(evaluation)
        st.session_state["current_question"] += 1
        # Clear audio bytes for next question
        st.session_state["audio_bytes"] = None
        st.experimental_rerun()

    st.info("Click Submit when you completed your answer.")

# --- Main app ---

def main():
    st.set_page_config(page_title="AI Interviewer with Voice", layout="wide", page_icon="🤖")
    with st.sidebar:
        st.image("https://cdn.pixabay.com/photo/2017/01/10/19/05/brain-1976510_1280.png", width=100)
        st.header("Settings & API Key")
        key_in = st.text_input("OpenAI API Key", type="password")
        st.session_state["openai_api_key"] = key_in
        st.markdown("---")

    # Step 1: Resume Input
    if "resume" not in st.session_state:
        resume = resume_upload_ui()
        if not resume:
            st.stop()
        st.session_state["resume"] = resume

    # Step 2: Candidate Info
    if "candidate_name" not in st.session_state:
        name, role, exp, num_qs, model = candidate_info_ui()
        if not name.strip():
            st.warning("Please enter candidate name.")
            st.stop()
        st.session_state["candidate_name"] = name.strip()
        st.session_state["role"] = role.strip()
        st.session_state["experience"] = exp
        st.session_state["num_questions"] = num_qs
        st.session_state["model"] = model

    # Step 3: Generate questions
    if "questions" not in st.session_state:
        with st.spinner("Generating interview questions..."):
            questions = generate_questions(
                st.session_state["resume"],
                st.session_state["role"],
                st.session_state["experience"],
                st.session_state["num_questions"],
                st.session_state["model"],
            )
        if not questions:
            st.error("Failed to generate interview questions.")
            st.stop()
        st.session_state["questions"] = questions

    # Step 4: Conduct interview with voice
    if "evaluations" not in st.session_state:
        st.session_state["evaluations"] = []

    evals = interview_ui_voice(st.session_state["questions"], st.session_state["resume"], st.session_state["model"])

    if evals and len(evals) == len(st.session_state["questions"]):
        # After finishing all questions show summary & export
        summary_ui(
            st.session_state["candidate_name"],
            st.session_state["role"],
            st.session_state["experience"],
            st.session_state["resume"],
            st.session_state["questions"],
            evals,
        )

if __name__ == "__main__":
    main()
