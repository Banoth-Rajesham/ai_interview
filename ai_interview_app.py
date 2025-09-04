"""
ai_interview.py
Streamlit AI Interviewer App without login/authentication,
with Resume Upload, Voice Q&A, GPT Evaluation, Reporting.
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
from streamlit_audio_recorder import st_audio_recorder


# --- Config ---
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
DB_PATH = "interview_sessions.db"
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# --- Helpers ---

def get_api_key():
    key = st.session_state.get("openai_api_key", "")
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            name TEXT,
            role TEXT,
            exp TEXT,
            date TEXT,
            session_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_session(user, name, role, exp, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (username, name, role, exp, date, session_json) VALUES (?, ?, ?, ?, ?, ?)",
             (user, name, role, exp, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(data)))
    conn.commit()
    conn.close()

def list_sessions(user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    res = c.execute("SELECT id, name, role, exp, date FROM sessions WHERE username=? ORDER BY date DESC LIMIT 50", (user,)).fetchall()
    conn.close()
    return res

def load_session(session_id, user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("SELECT session_json FROM sessions WHERE id=? AND username=?", (session_id, user)).fetchone()
    conn.close()
    if row: return json.loads(row[0])
    return None

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return clean_text(text)
    elif file.name.lower().endswith(".txt"):
        return clean_text(file.read().decode())
    else:
        st.error("Only PDF or TXT files supported.")
        return ""

def clean_text(txt):
    return "\n".join([line.strip() for line in txt.splitlines() if line.strip()])

def call_openai_chat(messages, model="gpt-4o", temperature=0.3, max_tokens=1200):
    key = get_api_key()
    if not key:
        st.error("Provide OpenAI API key in sidebar or environment.")
        st.stop()
    openai.api_key = key
    try:
        return openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

# Generate questions
def gen_questions(resume, role, exp, count, model):
    prompt = f"""
    You are a top interview coach. Based on this resume and role '{role}', experience '{exp}', generate exactly {count} interview questions.
    Return ONLY a JSON array of objects with id, text, topic, difficulty, estimated_time_seconds.
    Include 2+ personalized questions with reference to the resume.
    Resume:
    {resume}
    """
    msg = [{"role":"system","content":"You are a question generator."},{"role":"user","content":prompt}]
    resp = call_openai_chat(msg, model=model, max_tokens=2000)
    raw = resp.choices[0].message.content
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group())
            for i,q in enumerate(j): q['id'] = i+1
            return j
        except:
            try:
                j = ast.literal_eval(m.group())
                for i,q in enumerate(j): q['id'] = i+1
                return j
            except:
                st.error("Failed to parse questions JSON.")
                return []
    st.error("No question list found.")
    return []

# Evaluate answers
def evaluate(q, ans, resume, model):
    prompt = f"""
    You are an expert interviewer scoring this answer.
    Question: {q['text']}
    Candidate answer: {ans}
    Candidate resume: {resume}
    Score 0-10, justification, 3 improvements, model answer.
    Reply JSON only.
    """
    msg = [{"role":"system","content":"Score answers."},{"role":"user","content":prompt}]
    resp = call_openai_chat(msg, model=model, max_tokens=700)
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group())
            j['score'] = int(j.get('score',0))
            j['improvements'] = j.get('improvements',[])[:3]
            return j
        except:
            try:
                j = ast.literal_eval(m.group())
                j['score'] = int(j.get('score',0))
                j['improvements'] = j.get('improvements',[])[:3]
                return j
            except:
                return {"score":0,"justification":"Parse error.","improvements":[],"model_answer":""}
    return {"score":0,"justification":"Failed to evaluate.","improvements":[],"model_answer":""}

# Transcript audio
def transcribe_audio(audio_bytes, model="whisper-1"):
    key = get_api_key()
    if not key:
        st.error("OpenAI API key required for transcription.")
        return None
    openai.api_key = key
    with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
        tf.write(audio_bytes)
        tf.flush()
        try:
            res = openai.Audio.transcribe(model=model, file=open(tf.name, "rb"))
            return res.text
        except Exception as e:
            st.warning("Transcription failed: "+str(e))
            return None

# TTS
def tts_speak(text, voice="alloy"):
    key = get_api_key()
    if not key:
        st.warning("OpenAI API key required for TTS.")
        return None
    openai.api_key = key
    try:
        res = openai.Audio.speech.create(model="tts-1", voice=voice, input=text)
        return res.content
    except Exception as e:
        st.warning("TTS failed: "+str(e))
        return None

# PDF report
def generate_pdf(name, role, exp, qs, evals, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, 10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,10,"AI Interview Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0,8,f"Candidate: {name}", ln=True)
    pdf.cell(0,8,f"Role: {role}", ln=True)
    pdf.cell(0,8,f"Experience: {exp}", ln=True)
    pdf.cell(0,8,f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,8,"Interview Results", ln=True)
    pdf.set_font("Arial", "", 11)
    for i,q in enumerate(qs):
        e = evals[i] if i<len(evals) else {}
        pdf.multi_cell(0,8, f"Q{i+1}. {q['text']}")
        pdf.multi_cell(0,8,f"Topic: {q['topic']}, Difficulty: {q['difficulty']}")
        pdf.multi_cell(0,8, f"Answer: {e.get('answer','')}")
        pdf.multi_cell(0,8, f"Score: {e.get('score',0)}")
        pdf.multi_cell(0,8, f"Justification: {e.get('justification','')}")
        pdf.multi_cell(0,8, "Improvements: "+"; ".join(e.get('improvements',[])))
        pdf.multi_cell(0,8, f"Model Answer: {e.get('model_answer','')}")
        pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,8,"Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0,8, f"Overall Score: {summary.get('overall_score','-')}/10")
    pdf.multi_cell(0,8,"Strengths: "+ "; ".join(summary.get('strengths', [])))
    pdf.multi_cell(0,8,"Weaknesses: "+ "; ".join(summary.get('weaknesses', [])))
    pdf.multi_cell(0,8,f"Recommendation: {summary.get('recommendation','')}")
    pdf.multi_cell(0,8,"Next Steps: "+ "; ".join(summary.get('next_steps', [])))
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

def make_session_json(name, role, exp, resume, questions, evalu):
    summary = summarize_session(questions, evalu, resume, st.session_state.get('model_choice', "gpt-4o"))
    return {
        "candidate": {"name": name, "role": role, "exp": exp, "resume": resume},
        "interview": {
            "questions": questions,
            "evaluations": evalu,
            "summary": summary,
            "date": datetime.now().isoformat()
        }
    }

def summarize_session(questions, evaluations, resume, model):
    items = []
    for i, q in enumerate(questions):
        ev = evaluations[i] if i<len(evaluations) else {}
        items.append({
            "question": q['text'],
            "score": ev.get('score',0),
            "justification": ev.get('justification',''),
            "improvements": ev.get('improvements',[]),
            "answer": ev.get('answer',''),
            "model_answer": ev.get('model_answer','')
        })
    prompt = f"""
    You are an expert interviewer summarizing:
    {json.dumps(items,indent=2)}
    Resume:
    {resume}
    Return JSON with:
    overall_score (int 0-10), strengths [str], weaknesses [str], recommendation (Yes/No/Maybe), next_steps [str]
    """
    msgs = [{"role":"system","content":"You summarize interview results."},{"role":"user","content":prompt}]
    resp = call_openai_chat(msgs, model=model, max_tokens=600)
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group())
            j['overall_score'] = int(j.get('overall_score',5))
            return j
        except:
            return {"overall_score":5,"strengths":[],"weaknesses":[],"recommendation":"Maybe","next_steps":[]}
    return {"overall_score":5,"strengths":[],"weaknesses":[],"recommendation":"Maybe","next_steps":[]}

# --- UI parts ---

def sidebar_ui():
    st.sidebar.header("Settings & API Key")
    key = st.sidebar.text_input("OpenAI Key", type="password", help="Enter OpenAI API Key")
    st.session_state['openai_api_key'] = key.strip()
    st.sidebar.markdown("---")

def resume_upload_ui():
    st.header("Step 1: Upload Resume / Use Demo")
    f = st.file_uploader("Upload PDF or TXT Resume")
    use_demo = st.checkbox("Use Demo Resume")
    if f:
        try:
            txt = extract_text(f)
            st.success("Resume loaded")
            st.text_area("Preview:", txt, height=200)
            return txt
        except Exception as e:
            st.error("Resume parsing failed: "+str(e))
            return None
    if use_demo:
        st.info("Using demo resume")
        st.text_area("Demo Resume", DEFAULT_DEMO, height=200)
        return DEFAULT_DEMO
    st.warning("Upload resume or select demo to proceed")
    return None

def candidate_info_ui():
    st.header("Step 2: Candidate Details")
    name = st.text_input("Candidate Name")
    role = st.text_input("Position / Role","Software Engineer")
    exp = st.selectbox("Experience Level", ["Intern","Junior","Mid","Senior","Lead","Manager"], index=3)
    qcount = st.slider("No. of Questions", 3, 10, 5)
    model = st.selectbox("Model", list(MODELS.keys()), index=0)
    return name, role, exp, qcount, MODELS[model]

def interview_ui(questions, resume, model):
    st.header("Step 3: Interview")
    if 'current_q' not in st.session_state:
        st.session_state['current_q'] = 0
    idx = st.session_state['current_q']
    if idx >= len(questions):
        st.success("Interview completed")
        return st.session_state.get('evals', [])
    
    q = questions[idx]
    st.subheader(f"Question {idx+1} ({q['topic']} - {q['difficulty']})")
    st.write(q['text'])
    
    audio_bytes = None
    col1, col2 = st.columns([1,2])
    with col1:
        audio_bytes = st_audio_recorder(key=f"audio_{idx}", pause_threshold=2)
    with col2:
        ans_text = ""
        if audio_bytes:
            st.info("Transcribing audio...")
            ans_text = transcribe_audio(audio_bytes) or ""
            if ans_text:
                st.text_area("Your answer (transcribed)", ans_text, height=100, key=f"ans_{idx}")
            else:
                st.warning("Could not transcribe, please type your answer below.")
        ans_text = st.text_area("Your answer (type or overwrite)", ans_text, height=100, key=f"text_{idx}")

    if st.button("Submit Answer", key=f"submit_{idx}"):
        if not ans_text.strip():
            st.warning("Answer cannot be empty.")
            st.stop()
        with st.spinner("Evaluating..."):
            ev = evaluate(q, ans_text.strip(), resume, model)
            ev['answer'] = ans_text.strip()
        if 'evals' not in st.session_state:
            st.session_state['evals'] = []
        st.session_state['evals'].append(ev)
        st.success(f"Score: {ev['score']}/10")
        st.write(f"Justification: {ev['justification']}")
        st.write("Improvements:")
        for imp in ev['improvements']:
            st.write(f"- {imp}")
        st.write(f"Model answer: {ev['model_answer']}")
        st.session_state['current_q'] += 1
        st.experimental_rerun()
    else:
        st.info("Press submit when ready.")

def summary_ui(name, role, exp, resume, questions, evals):
    st.header("Step 4: Summary and Export")
    summ = summarize_session(questions, evals, resume, st.session_state.get('model_choice',"gpt-4o"))
    st.write(f"Overall Score: {summ.get('overall_score','-')}/10")
    st.write("Strengths: "+", ".join(summ.get('strengths',[])))
    st.write("Weaknesses: "+", ".join(summ.get('weaknesses',[])))
    st.write(f"Recommendation: {summ.get('recommendation','')}")
    st.write("Next Steps: "+ ", ".join(summ.get('next_steps',[])))

    pdf_buf = generate_pdf(name, role, exp, questions, evals, summ)
    st.download_button("Download PDF Report", pdf_buf,
                       file_name=f"{name}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                       mime='application/pdf')

    session_json = make_session_json(name, role, exp, resume, questions, evals)
    json_bytes = json.dumps(session_json, indent=2).encode()
    st.download_button("Export JSON Session", json_bytes,
                       file_name=f"{name}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                       mime='application/json')

    if st.button("Save Session"):
        save_session("guest", name, role, exp, session_json)
        filepath = os.path.join(SESSION_DIR, f"{name}_Interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filepath, "w") as f:
            json.dump(session_json, f, indent=2)
        st.success("Session saved! Please restart app for new session.")
        for k in ['current_q','evals','questions','resume','candidate_name','role','exp']:
            if k in st.session_state: del st.session_state[k]
        st.stop()

# --- Main ---
def main():
    st.set_page_config("AI Interviewer", layout="wide")
    sidebar_ui()
    init_db()

    if 'questions' not in st.session_state:
        resume = resume_upload_ui()
        if not resume:
            st.stop()
        st.session_state['resume'] = resume
        name, role, exp, qcount, model = candidate_info_ui()
        st.session_state['candidate_name'] = name
        st.session_state['role'] = role
        st.session_state['exp'] = exp
        st.session_state['model_choice'] = model

        if not name.strip():
            st.warning("Please enter candidate name.")
            st.stop()

        with st.spinner("Generating questions..."):
            qs = gen_questions(resume, role, exp, qcount, model)
            st.session_state['questions'] = qs
            st.session_state['evals'] = []
            st.session_state['current_q'] = 0

    else:
        qs = st.session_state['questions']
        evals = interview_ui(qs, st.session_state['resume'], st.session_state['model_choice'])
        if evals and len(evals) == len(qs):
            summary_ui(st.session_state['candidate_name'], st.session_state['role'], st.session_state['exp'], st.session_state['resume'], qs, evals)

if __name__ == "__main__":
    main()
