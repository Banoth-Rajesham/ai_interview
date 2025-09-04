"""
ai_interview_app.py

Streamlit AI Interviewer App with Login/Auth, Resume Upload, Voice Q&A, GPT Evaluation, and Reporting

Dependencies:
- streamlit
- openai
- PyPDF2
- fpdf
- numpy
- sqlite3
- streamlit-authenticator

Setup:
1. pip install -r requirements.txt
2. Run:
   streamlit run ai_interview_app.py
3. Provide your OpenAI API Key in the sidebar or set as an environment variable.

Features:
- User login / registration (basic, demo accounts)
- Resume upload or demo mode
- GPT-generated interview questions (personalized)
- Voice playback and speech-to-text answers (Whisper API) with fallback to text
- GPT scoring, feedback, model answers
- Downloadable PDF report and JSON export
- Session saving in SQLite and JSON

Author: Perplexity AI
"""

import streamlit as st
import openai
import PyPDF2
import tempfile
import io
import json
import sqlite3
import numpy as np
from fpdf import FPDF
from datetime import datetime
import os
import re
import ast
import streamlit_authenticator as stauth

# === Config === #
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
DEFAULT_DEMO_RESUME = """
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
DB_PATH = "interview_sessions.db"
SESSION_JSON_DIR = "saved_sessions"
os.makedirs(SESSION_JSON_DIR, exist_ok=True)

# === Helpers: API Key === #
def get_openai_api_key():
    key = st.session_state.get("openai_api_key", "")
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key

# === Helpers: DB === #
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        name TEXT,
        role TEXT,
        exp_level TEXT,
        date TEXT,
        session_json TEXT
    )""")
    conn.commit()
    conn.close()

def save_session_to_db(username, name, role, exp_level, session_json):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions (username, name, role, exp_level, date, session_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, name, role, exp_level, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(session_json)))
    conn.commit()
    conn.close()

def list_sessions_from_db(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    results = c.execute("SELECT id, name, role, exp_level, date FROM sessions WHERE username=? ORDER BY date DESC LIMIT 50", (username,)).fetchall()
    conn.close()
    return results

def get_session_json_from_db(session_id, username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    result = c.execute("SELECT session_json FROM sessions WHERE id=? AND username=?", (session_id, username)).fetchone()
    conn.close()
    return json.loads(result[0]) if result else None

# === Helpers: Resume Parsing === #
def extract_text_from_resume(uploaded_file):
    if uploaded_file.name.lower().endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return clean_resume_text(text)
    elif uploaded_file.name.lower().endswith('.txt'):
        text = uploaded_file.read().decode("utf-8")
        return clean_resume_text(text)
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT allowed.")

def clean_resume_text(text):
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# === Helpers: OpenAI API calls === #
def openai_chat(messages, model="gpt-4o", temperature=0.3, max_tokens=1200):
    openai.api_key = get_openai_api_key()
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

def generate_interview_questions(resume_text, role, exp_level, question_count, model_name):
    prompt = f"""
You are a top interview coach. Based on this resume, role '{role}', experience '{exp_level}', generate exactly {question_count} interview questions.
Each question should have JSON fields: id (int), text, topic, difficulty (Easy/Medium/Hard), estimated_time_seconds.
Include at least 2 personalized questions from the resume content.

Resume:
{resume_text}
Return ONLY a JSON array of question objects.
"""
    messages = [
        {"role": "system", "content": "You are a question generating assistant."},
        {"role": "user", "content": prompt}
    ]
    resp = openai_chat(messages, model=model_name, max_tokens=2000)
    raw = resp.choices[0].message.content
    match = re.search(r"\[\s*{.*}\s*\]", raw, re.DOTALL)
    if match:
        q_array = match.group(0)
        try:
            data = json.loads(q_array)
            for i, q in enumerate(data):
                q["id"] = i+1
            return data
        except Exception:
            try:
                data = ast.literal_eval(q_array)
                for i, q in enumerate(data):
                    q["id"] = i+1
                return data
            except:
                st.error("Unable to parse questions JSON from GPT. Try again.")
                return []
    else:
        st.error("No questions JSON found in GPT output. Try again.")
        return []

def evaluate_answer(question, answer, resume, model_name):
    prompt = f"""
You are an expert interviewer scoring this answer:

Question: {question['text']}
Candidate's answer: {answer}
Role: {st.session_state.get('role', '')}
Experience Level: {st.session_state.get('exp_level', '')}
Candidate Resume: {resume}

Score (0-10), justification, 3 improvements, model answer.
Return JSON only in format:
{{
"score": integer,
"justification": string,
"improvements": [string, string, string],
"model_answer": string
}}
"""
    messages = [
        {"role": "system", "content": "You evaluate interview answers strictly."},
        {"role": "user", "content": prompt}
    ]
    resp = openai_chat(messages, model=model_name, max_tokens=700)
    raw = resp.choices[0].message.content
    match = re.search(r"{.*}", raw, re.DOTALL)
    if match:
        try:
            eval_json = json.loads(match.group(0))
            eval_json['score'] = int(eval_json.get('score', 0))
            eval_json['improvements'] = eval_json.get('improvements', [])[:3]
            return eval_json
        except Exception:
            try:
                eval_json = ast.literal_eval(match.group(0))
                eval_json['score'] = int(eval_json.get('score', 0))
                eval_json['improvements'] = eval_json.get('improvements', [])[:3]
                return eval_json
            except:
                return {"score": 0, "justification": "Parse failure.", "improvements": [], "model_answer": ""}
    else:
        return {"score": 0, "justification": "Evaluation failed.", "improvements": [], "model_answer": ""}

def summarize_session(questions, answers_evals, resume, model_name):
    items = []
    for i, q in enumerate(questions):
        e = answers_evals[i] if i < len(answers_evals) else {}
        items.append({
            "question": q["text"],
            "score": e.get("score", 0),
            "justification": e.get("justification", ""),
            "improvements": e.get("improvements", []),
            "answer": e.get("answer", ""),
            "model_answer": e.get("model_answer", "")
        })
    prompt = f"""
You are an expert interviewer summarizing interview results:

{json.dumps(items, indent=2)}

Candidate Resume:
{resume}

Return JSON only with:
{{
  "overall_score": 0-10 integer,
  "strengths": [str],
  "weaknesses": [str],
  "recommendation": "Yes/No/Maybe",
  "next_steps": [str]
}}
"""
    messages = [
        {"role": "system", "content": "You summarize interview outcomes strictly."},
        {"role": "user", "content": prompt}
    ]
    resp = openai_chat(messages, model=model_name, max_tokens=600)
    raw = resp.choices[0].message.content
    match = re.search(r"{.*}", raw, re.DOTALL)
    if match:
        try:
            summary = json.loads(match.group(0))
            summary["overall_score"] = int(summary.get("overall_score", 5))
            return summary
        except:
            return {
                "overall_score": 5,
                "strengths": [],
                "weaknesses": [],
                "recommendation": "Maybe",
                "next_steps": []
            }
    else:
        return {
            "overall_score": 5,
            "strengths": [],
            "weaknesses": [],
            "recommendation": "Maybe",
            "next_steps": []
        }

# Whisper API STT
def transcribe_audio(audio_bytes, model="whisper-1"):
    openai.api_key = get_openai_api_key()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        tmp_wav.write(audio_bytes)
        tmp_wav.flush()
        try:
            transcript = openai.Audio.transcribe(model=model, file=open(tmp_wav.name, "rb"))
            return transcript.text
        except Exception:
            st.warning("Speech-to-Text failed. Please type your answer.")
            return None

# TTS playback via OpenAI TTS API (if available)
def tts_speak(text, voice="alloy"):
    openai.api_key = get_openai_api_key()
    try:
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
        return response.content
    except Exception:
        st.warning("Text-to-Speech failed. Please read the question yourself.")
        return None

# === Helpers: PDF report generation === #
def generate_pdf_report(candidate_name, role, exp_level, questions, answers_evals, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "AI Interview Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Candidate: {candidate_name}", ln=True)
    pdf.cell(0, 8, f"Role: {role}", ln=True)
    pdf.cell(0, 8, f"Experience Level: {exp_level}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Interview Results", ln=True)
    pdf.set_font("Arial", size=11)

    for i, q in enumerate(questions):
        e = answers_evals[i] if i < len(answers_evals) else {}
        pdf.cell(0, 8, f"Q{i+1}. {q['text']}", ln=True)
        pdf.cell(0, 8, f" Topic: {q['topic']}, Difficulty: {q['difficulty']}", ln=True)
        pdf.multi_cell(0, 8, f" Answer: {e.get('answer', '')}", ln=True)
        pdf.multi_cell(0, 8, f" Score: {e.get('score', 0)}")
        pdf.multi_cell(0, 8, f" Justification: {e.get('justification', '')}")
        pdf.multi_cell(0, 8, " Improvements: " + "; ".join(e.get('improvements', [])))
        pdf.multi_cell(0, 8, f" Model Answer: {e.get('model_answer', '')}")
        pdf.ln(3)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Overall Score: {summary.get('overall_score', '-')}/10", ln=True)
    pdf.cell(0, 8, "Strengths: " + "; ".join(summary.get('strengths', [])), ln=True)
    pdf.cell(0, 8, "Weaknesses: " + "; ".join(summary.get('weaknesses', [])), ln=True)
    pdf.cell(0, 8, f"Recommendation: {summary.get('recommendation', '')}", ln=True)
    pdf.cell(0, 8, "Next Steps: " + "; ".join(summary.get('next_steps', [])), ln=True)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# === Helpers: Session JSON export === #
def make_session_json(candidate_name, role, exp_level, resume, questions, answers_evals, summary):
    return {
        "candidate": {"name": candidate_name, "role": role, "exp_level": exp_level, "resume": resume},
        "interview": {
            "date": datetime.now().isoformat(),
            "questions": questions,
            "answers_evals": answers_evals,
            "summary": summary,
        }
    }

# === UI Components === #
def ui_sidebar(username):
    st.sidebar.header("Settings & API Key")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=get_openai_api_key(), help="Enter your OpenAI API Key")
    st.session_state["openai_api_key"] = openai_key.strip()

    st.sidebar.markdown("---")
    st.sidebar.write(f"Logged in as: **{username}**")

    if st.sidebar.button("Show Previous Sessions"):
        sessions = list_sessions_from_db(username)
        if not sessions:
            st.sidebar.info("No saved sessions available")
        else:
            for id_, name, role, exp, date in sessions:
                st.sidebar.write(f"ID: {id_} | {name} | {role} | Exp: {exp} | {date}")
            load_id = st.sidebar.text_input("Enter Session ID to Load")
            if load_id:
                try:
                    j = get_session_json_from_db(int(load_id), username)
                    st.session_state["loaded_session_json"] = j
                    st.sidebar.success("Session loaded!")
                except:
                    st.sidebar.error("Failed to load session. Check ID or try again.")

def ui_resume_upload():
    st.subheader("Step 1: Upload Resume or Use Demo")
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    use_demo = st.checkbox("Use Demo Resume")
    if uploaded_file:
        try:
            resume_text = extract_text_from_resume(uploaded_file=uploaded_file)
            st.success("Resume successfully uploaded and parsed.")
            st.text_area("Resume Preview", resume_text, height=180)
            return resume_text
        except Exception as e:
            st.error("Failed to parse resume: " + str(e))
            return None
    if use_demo:
        st.info("Demo resume loaded.")
        st.text_area("Demo Resume Preview", DEFAULT_DEMO_RESUME, height=180)
        return DEFAULT_DEMO_RESUME
    st.warning("Please upload a resume file or select demo resume.")
    return None

def ui_candidate_info():
    st.subheader("Step 2: Enter Candidate Details")
    candidate_name = st.text_input("Candidate Name")
    role = st.text_input("Position / Role", value="Software Engineer")
    exp_level = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager"])
    question_num = st.slider("Number of Interview Questions", 3, 10, 5)
    model_choice = st.selectbox("OpenAI Model", list(MODELS.keys()), index=0)
    return candidate_name, role, exp_level, question_num, MODELS[model_choice]

def ui_instructions():
    st.subheader("Instructions")
    st.markdown("""
    - The system will generate personalized interview questions based on your resume.
    - Each question will be read aloud.
    - Please answer using your microphone; if speech recognition fails, type your answer.
    - After all questions, receive detailed evaluation, score, and suggestions.
    - Export full report as PDF or JSON.
    """)

def ui_interview(questions, resume_text, model_name):
    st.subheader("Step 3: Interview In Progress")
    st.write(f"Role: {st.session_state.get('role','')}, Model: {model_name}, Questions: {len(questions)}")
    
    answers_evaluations = []
    for idx, question in enumerate(questions):
        st.markdown(f"**Q{idx+1}. ({question['topic']}, {question['difficulty']}, ~{question['estimated_time_seconds']}s)**")
        st.write(question['text'])

        tts_audio = tts_speak(question['text'])
        col1, col2 = st.columns([1, 2])
        if tts_audio:
            with col1:
                st.audio(tts_audio, format="audio/mp3")
        else:
            with col1:
                st.info("Cannot play audio question. Please read it.")

        with col2:
            st.write("Answer via microphone:")
            audio_input = st.audio_recorder(key=f"mic_{idx}", pause_threshold=2)
            candidate_answer = None
            if audio_input:
                candidate_answer = transcribe_audio(audio_input)
            if not candidate_answer:
                candidate_answer = st.text_area(f"Or type your answer (Q{idx+1})", key=f"text_{idx}")
            if not candidate_answer:
                st.warning("Awaiting your answer...")
                st.stop()

            st.info("Evaluating your answer...")
            eval_result = evaluate_answer(question, candidate_answer, resume_text, model_name)
            eval_result['answer'] = candidate_answer
            st.success(f"Score: {eval_result['score']}/10")
            st.markdown(f"**Justification:** {eval_result['justification']}")
            st.markdown("**Improvements:**")
            for imp in eval_result['improvements']:
                st.write(f"- {imp}")
            st.markdown(f"**Model Answer:** {eval_result['model_answer']}")
            answers_evaluations.append(eval_result)
            st.markdown("---")
    return answers_evaluations

def ui_summary_and_export(username, candidate_name, role, exp_level, resume_text, questions, answers_evals, summary):
    st.header("Interview Summary & Report")
    st.subheader("Aggregate Results")
    st.write(f"Overall Score: {summary.get('overall_score', '-')}/10")
    st.write("Strengths: " + "; ".join(summary.get('strengths', [])))
    st.write("Weaknesses: " + "; ".join(summary.get('weaknesses', [])))
    st.write(f"Recommendation: {summary.get('recommendation', '')}")
    st.write("Suggested Next Steps: " + "; ".join(summary.get('next_steps', [])))
    st.markdown("---")
    st.subheader("Download Artifacts")

    pdf_buf = generate_pdf_report(candidate_name, role, exp_level, questions, answers_evals, summary)
    st.download_button("Download PDF Report", data=pdf_buf, file_name=f"Interview_Report_{candidate_name}_{datetime.now().strftime('%Y%m%d')}.pdf")
    
    session_json = make_session_json(candidate_name, role, exp_level, resume_text, questions, answers_evals, summary)
    st.download_button("Export Session JSON", data=json.dumps(session_json, indent=2), file_name=f"Interview_Session_{candidate_name}_{datetime.now().strftime('%Y%m%d')}.json")

    if st.button("Save Session to DB and JSON Folder"):
        save_session_to_db(username, candidate_name, role, exp_level, session_json)
        with open(f"{SESSION_JSON_DIR}/Interview_Session_{candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            f.write(json.dumps(session_json, indent=2))
        st.success("Session saved!")

# === Authentication Setup ===
def auth_login():
    names = ["Alice", "Bob", "Carol"]
    usernames = ["alice", "bob", "carol"]
    # Pre-generated password hashes for "123", "abc", "xyz"
    passwords = [
        '$2b$12$CAZ4rUYKp3RKMzuiFKGPO.FFVWSSxa6ZPGnaAulyi6iryVyU8uiIS',
        '$2b$12$yCFD8Pv/WGS0LvpGBOSmyuWT.ZMJxw7AgtbKaaOyakexwMRglTcSO',
        '$2b$12$WWMNcfbXUgBraQjbqoSLse6RuNRCUnhotv5PWmlW.S7KBoZ2ziOYq'
    ]
    authenticator = stauth.Authenticate(
        names, usernames, passwords,
        cookie_name="interview_app_cookie",
        key="interview_app_signature",
        cookie_expiry_days=30
    )
    name, auth_status, username = authenticator.login("Login", "main")
    if auth_status:
        st.session_state["username"] = username
        return authenticator, username
    elif auth_status is False:
        st.error("Username/password is incorrect")
    elif auth_status is None:
        st.warning("Please enter your credentials")
    return authenticator, None

# === Main App === #
def main():
    st.set_page_config(page_title="AI Interviewer App", layout="wide")
    init_db()
    authenticator, username = auth_login()
    if not username:
        return
    ui_sidebar(username)

    st.title("🔊 AI Interviewer with AI-powered Q&A and Evaluation")
    
    # Step 1: Resume Upload / Demo
    resume_text = ui_resume_upload()
    if not resume_text:
        st.stop()

    # Step 2: Candidate Info
    candidate_name, role, exp_level, question_count, model_name = ui_candidate_info()
    if not candidate_name:
        st.warning("Please enter your candidate name to proceed.")
        st.stop()

    st.session_state['role'] = role
    st.session_state['exp_level'] = exp_level

    st.markdown("---")
    ui_instructions()

    start_btn = st.button("Start Interview")
    if start_btn:
        st.info("Generating questions...")
        questions = generate_interview_questions(resume_text, role, exp_level, question_count, model_name)
        if not questions:
            st.error("Question generation failed. Please try again.")
            st.stop()

        answers_evals = ui_interview(questions, resume_text, model_name)
        st.info("Generating summary...")
        summary = summarize_session(questions, answers_evals, resume_text, model_name)
        
        ui_summary_and_export(username, candidate_name, role, exp_level, resume_text, questions, answers_evals, summary)

    elif "loaded_session_json" in st.session_state:
        loaded = st.session_state["loaded_session_json"]
        st.header("Loaded Previous Interview Session")
        ui_summary_and_export(
            username,
            loaded["candidate"]["name"],
            loaded["candidate"]["role"],
            loaded["candidate"]["exp_level"],
            loaded["candidate"]["resume"],
            loaded["interview"]["questions"],
            loaded["interview"]["answers_evals"],
            loaded["interview"]["summary"]
        )

if __name__ == "__main__":
    main()
