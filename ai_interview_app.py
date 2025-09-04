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
- streamlit-audio-recorder (added for clearer dependency)

Setup:
1. pip install streamlit openai PyPDF2 fpdf numpy sqlite3 streamlit-authenticator streamlit-audio-recorder
2. Run:
   streamlit run ai_interview_app.py
3. Provide your OpenAI API Key in the sidebar or set as an environment variable (OPENAI_API_KEY).

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
import numpy as np # Although numpy isn't directly used in the provided code, it's a common dependency for data science apps
from fpdf import FPDF
from datetime import datetime
import os
import re
import ast
import streamlit_authenticator as stauth
from streamlit_audio_recorder import st_audio_recorder # Added for explicit audio recording widget

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
    if not openai.api_key:
        st.error("OpenAI API key not found. Please provide it in the sidebar.")
        st.stop()
    try:
        # Use ChatCompletion.create for older OpenAI library versions, or client.chat.completions.create for newer
        # This code assumes an older version if openai.ChatCompletion is directly available.
        # If using newer `openai` library (>=1.0.0), you'd need `client = openai.OpenAI(); client.chat.completions.create(...)`
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        st.stop()


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
                # Fallback to literal_eval for malformed JSON that Python can interpret
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
                # Fallback for malformed JSON
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
    if not openai.api_key:
        st.error("OpenAI API key not found for speech-to-text. Please provide it in the sidebar.")
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        tmp_wav.write(audio_bytes)
        tmp_wav.flush()
        try:
            # For newer OpenAI library (>=1.0.0), you'd need `client = openai.OpenAI(); client.audio.transcriptions.create(...)`
            transcript = openai.Audio.transcribe(model=model, file=open(tmp_wav.name, "rb"))
            return transcript.text
        except openai.error.OpenAIError as e:
            st.warning(f"Speech-to-Text failed due to OpenAI API error: {e}. Please type your answer.")
            return None
        except Exception as e:
            st.warning(f"Speech-to-Text failed: {e}. Please type your answer.")
            return None

# TTS playback via OpenAI TTS API (if available)
def tts_speak(text, voice="alloy"):
    openai.api_key = get_openai_api_key()
    if not openai.api_key:
        st.warning("OpenAI API key not found for text-to-speech. Please read the question yourself.")
        return None
    try:
        # For newer OpenAI library (>=1.0.0), you'd need `client = openai.OpenAI(); client.audio.speech.create(...)`
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
        return response.content
    except openai.error.OpenAIError as e:
        st.warning(f"Text-to-Speech failed due to OpenAI API error: {e}. Please read the question yourself.")
        return None
    except Exception as e:
        st.warning(f"Text-to-Speech failed: {e}. Please read the question yourself.")
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
        pdf.multi_cell(0, 8, f"Q{i+1}. {q['text']}")
        pdf.multi_cell(0, 8, f" Topic: {q['topic']}, Difficulty: {q['difficulty']}")
        pdf.multi_cell(0, 8, f" Answer: {e.get('answer', '')}")
        pdf.multi_cell(0, 8, f" Score: {e.get('score', 0)}")
        pdf.multi_cell(0, 8, f" Justification: {e.get('justification', '')}")
        pdf.multi_cell(0, 8, " Improvements: " + "; ".join(e.get('improvements', [])))
        pdf.multi_cell(0, 8, f" Model Answer: {e.get('model_answer', '')}")
        pdf.ln(3)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, f"Overall Score: {summary.get('overall_score', '-')}/10")
    pdf.multi_cell(0, 8, "Strengths: " + "; ".join(summary.get('strengths', [])))
    pdf.multi_cell(0, 8, "Weaknesses: " + "; ".join(summary.get('weaknesses', [])))
    pdf.multi_cell(0, 8, f"Recommendation: {summary.get('recommendation', '')}")
    pdf.multi_cell(0, 8, "Next Steps: " + "; ".join(summary.get('next_steps', [])))

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

    # This part needs a clear distinction between displaying and acting
    if st.sidebar.button("Show Previous Sessions", key="show_sessions_btn"):
        st.session_state["show_past_sessions"] = True

    if st.session_state.get("show_past_sessions", False):
        sessions = list_sessions_from_db(username)
        if not sessions:
            st.sidebar.info("No saved sessions available")
        else:
            st.sidebar.subheader("Available Sessions")
            session_options = [f"ID: {id_} | {name} | {role} | Exp: {exp} | {date}" for id_, name, role, exp, date in sessions]
            selected_session_display = st.sidebar.selectbox("Select Session to Load", [""] + session_options, key="select_session_to_load")

            if selected_session_display:
                session_id = int(selected_session_display.split(" | ")[0].replace("ID: ", ""))
                try:
                    j = get_session_json_from_db(session_id, username)
                    st.session_state["loaded_session_json"] = j
                    st.sidebar.success(f"Session ID {session_id} loaded!")
                    st.session_state["show_past_sessions"] = False # Hide after loading
                    st.rerun() # Rerun to display loaded session immediately
                except Exception as e:
                    st.sidebar.error(f"Failed to load session {session_id}. Error: {e}")
            st.sidebar.markdown("---")


def ui_resume_upload():
    st.subheader("Step 1: Upload Resume or Use Demo")
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    use_demo = st.checkbox("Use Demo Resume", key="use_demo_resume")
    
    resume_text = None
    if uploaded_file:
        try:
            resume_text = extract_text_from_resume(uploaded_file=uploaded_file)
            st.success("Resume successfully uploaded and parsed.")
            st.text_area("Resume Preview", resume_text, height=180, key="uploaded_resume_preview")
        except Exception as e:
            st.error("Failed to parse resume: " + str(e))
    elif use_demo:
        st.info("Demo resume loaded.")
        resume_text = DEFAULT_DEMO_RESUME
        st.text_area("Demo Resume Preview", DEFAULT_DEMO_RESUME, height=180, key="demo_resume_preview")

    if not resume_text:
        st.warning("Please upload a resume file or select demo resume to proceed.")
        return None
    return resume_text

def ui_candidate_info():
    st.subheader("Step 2: Enter Candidate Details")
    candidate_name = st.text_input("Candidate Name", key="candidate_name")
    role = st.text_input("Position / Role", value="Software Engineer", key="candidate_role")
    exp_level = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager"], key="exp_level_select")
    question_num = st.slider("Number of Interview Questions", 3, 10, 5, key="question_count_slider")
    model_choice = st.selectbox("OpenAI Model", list(MODELS.keys()), index=0, key="model_select")
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
    # Ensure current_question_idx is initialized
    if "current_question_idx" not in st.session_state:
        st.session_state["current_question_idx"] = 0

    if st.session_state["current_question_idx"] < len(questions):
        idx = st.session_state["current_question_idx"]
        question = questions[idx]

        st.markdown(f"**Q{idx+1}. ({question['topic']}, {question['difficulty']}, ~{question['estimated_time_seconds']}s)**")
        st.write(question['text'])

        tts_audio_content = tts_speak(question['text'])
        col1, col2 = st.columns([1, 2])
        if tts_audio_content:
            with col1:
                st.audio(tts_audio_content, format="audio/mp3", autoplay=True, key=f"tts_audio_{idx}")
        else:
            with col1:
                st.info("Cannot play audio question. Please read it.")

        with col2:
            st.write("Answer via microphone:")
            # Use the explicit st_audio_recorder
            audio_bytes = st_audio_recorder(key=f"audio_recorder_{idx}", pause_threshold=2)
            candidate_answer = None

            if audio_bytes:
                st.info("Transcribing audio...")
                candidate_answer = transcribe_audio(audio_bytes)
                if candidate_answer:
                    st.text_area("Your transcribed answer:", value=candidate_answer, height=100, key=f"transcribed_answer_{idx}")
                else:
                    st.warning("Could not transcribe audio. Please type your answer below.")
            
            # Fallback to text area if no audio or transcription failed
            typed_answer = st.text_area(f"Or type your answer (Q{idx+1})", value=candidate_answer if candidate_answer else "", key=f"text_answer_{idx}")
            
            final_answer = typed_answer.strip()

            if st.button(f"Submit Answer for Q{idx+1}", key=f"submit_q_{idx}"):
                if not final_answer:
                    st.warning("Please provide an answer before submitting.")
                    st.stop()
                
                with st.spinner("Evaluating your answer..."):
                    eval_result = evaluate_answer(question, final_answer, resume_text, model_name)
                    eval_result['answer'] = final_answer
                
                # Store evaluation result temporarily
                if "all_answers_evals" not in st.session_state:
                    st.session_state["all_answers_evals"] = []
                st.session_state["all_answers_evals"].append(eval_result)

                st.success(f"Score: {eval_result['score']}/10")
                st.markdown(f"**Justification:** {eval_result['justification']}")
                st.markdown("**Improvements:**")
                for imp in eval_result['improvements']:
                    st.write(f"- {imp}")
                st.markdown(f"**Model Answer:** {eval_result['model_answer']}")
                st.markdown("---")

                st.session_state["current_question_idx"] += 1
                st.rerun() # Rerun to display the next question or summary
            else:
                st.info("Submit your answer to proceed.")
                st.stop() # Halt execution until submit button is pressed
    else:
        # All questions answered
        st.success("You have completed the interview!")
        answers_evaluations = st.session_state.get("all_answers_evals", [])
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
    st.download_button("Download PDF Report", data=pdf_buf, file_name=f"Interview_Report_{candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    
    session_json = make_session_json(candidate_name, role, exp_level, resume_text, questions, answers_evals, summary)
    st.download_button("Export Session JSON", data=json.dumps(session_json, indent=2), file_name=f"Interview_Session_{candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    if st.button("Save Session to DB and JSON Folder", key="save_session_btn"):
        save_session_to_db(username, candidate_name, role, exp_level, session_json)
        # Ensure the directory exists before writing
        os.makedirs(SESSION_JSON_DIR, exist_ok=True)
        with open(f"{SESSION_JSON_DIR}/Interview_Session_{candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            f.write(json.dumps(session_json, indent=2))
        st.success("Session saved!")
        # Clear session state to allow starting a new interview cleanly
        for key in ["current_question_idx", "all_answers_evals", "questions_generated", "loaded_session_json"]:
            if key in st.session_state:
                del st.session_state[key]
        st.info("You can now start a new interview or view saved sessions.")

# === Authentication Setup ===
def auth_login():
    # Define credentials in new format
    credentials = {
        "usernames": {
            "alice": {
                "name": "Alice",
                "password": stauth.Hasher(['123']).generate()[0] # Hash passwords once if not already hashed
            },
            "bob": {
                "name": "Bob",
                "password": stauth.Hasher(['abc']).generate()[0]
            },
            "carol": {
                "name": "Carol",
                "password": stauth.Hasher(['xyz']).generate()[0]
            }
        }
    }
    # It's better to hash passwords once and store them.
    # For demo, I'm doing it here, but in a real app, you'd store hashed passwords.
    # The existing hashed passwords are fine if they came from an earlier run.
    # You can get the hash for a password like this in a separate script:
    # import streamlit_authenticator as stauth
    # hashed_passwords = stauth.Hasher(['your_password']).generate()
    # print(hashed_passwords[0])

    # The existing hashed passwords in your provided code were correct.
    # I'm using the Hasher here for demonstration of generating them.
    # You can keep the hardcoded hashed strings if they are what you intend to use.
    credentials_to_use = {
        "usernames": {
            "alice": {
                "name": "Alice",
                "password": "$2b$12$CAZ4rUYKp3RKMzuiFKGPO.FFVWSSxa6ZPGnaAulyi6iryVyU8uiIS"
            },
            "bob": {
                "name": "Bob",
                "password": "$2b$12$yCFD8Pv/WGS0LvpGBOSmyuWT.ZMJxw7AgtbKaaOyakexwMRglTcSO"
            },
            "carol": {
                "name": "Carol",
                "password": "$2b$12$WWMNcfbXUgBraQjbqoSLse6RuNRCUnhotv5PWmlW.S7KBoZ2ziOYq"
            }
        }
    }


    authenticator = stauth.Authenticate(
        credentials_to_use, # Use the pre-hashed credentials
        "interview_app_cookie",
        "interview_app_signature",
        cookie_expiry_days=30
    )

    # Correct call for streamlit-authenticator login
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

    # Authentication must happen before any other Streamlit components are rendered
    authenticator, username = auth_login()
    
    if not username:
        # If not logged in, stop execution after login component
        st.stop()

    # User is logged in, proceed with the app
    ui_sidebar(username)

    st.title("🔊 AI Interviewer with AI-powered Q&A and Evaluation")
    
    # Check if a session was loaded from the sidebar
    if "loaded_session_json" in st.session_state and st.session_state["loaded_session_json"]:
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
        # Clear the loaded session after displaying to prevent it from re-displaying on rerun
        st.session_state["loaded_session_json"] = None
        return # Stop execution after displaying loaded session

    # Initialize session state variables for the interview process
    if "questions_generated" not in st.session_state:
        st.session_state["questions_generated"] = []
    if "all_answers_evals" not in st.session_state:
        st.session_state["all_answers_evals"] = []
    if "current_question_idx" not in st.session_state:
        st.session_state["current_question_idx"] = 0


    # Step 1: Resume Upload / Demo
    resume_text = ui_resume_upload()
    if not resume_text:
        st.stop() # Halt until resume is provided

    # Step 2: Candidate Info
    candidate_name, role, exp_level, question_count, model_name = ui_candidate_info()
    if not candidate_name:
        st.warning("Please enter your
