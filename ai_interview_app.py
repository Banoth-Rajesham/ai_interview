"""
ai_interview_noauth.py
Streamlit AI Interviewer App without login,
without audio recording (text input only),
with GPT evaluation, PDF and JSON export.
"""

import streamlit as st
import openai
import PyPDF2
import tempfile
import io
import json
import os
from fpdf import FPDF
from datetime import datetime
import re
import ast

# === Config ===
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

# === Helpers ===

def get_api_key():
    key = st.session_state.get("openai_api_key", "")
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key

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
        st.error("Only PDF and TXT files supported.")
        return ""

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def call_openai_chat(messages, model="gpt-4o", temperature=0.3, max_tokens=1200):
    key = get_api_key()
    if not key:
        st.error("Please provide your OpenAI API Key!")
        st.stop()
    openai.api_key = key
    try:
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

# Generate interview questions
def gen_questions(resume, role, exp, count, model):
    prompt = f"""
You are an expert interviewer. Based on this resume and the role '{role}' with experience '{exp}', generate exactly {count} interview questions in JSON format.
Include personalized questions referencing the resume.
Return a JSON array with fields: id, text, topic, difficulty, estimated_time_seconds.
Resume:
{resume}
"""
    messages = [
        {"role": "system", "content": "You generate interview questions."},
        {"role": "user", "content": prompt}
    ]
    resp = call_openai_chat(messages, model=model, max_tokens=2000)
    raw = resp.choices[0].message.content
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group())
            for i, q in enumerate(questions):
                q["id"] = i + 1
            return questions
        except:
            try:
                questions = ast.literal_eval(match.group())
                for i, q in enumerate(questions):
                    q["id"] = i + 1
                return questions
            except:
                st.error("Failed to parse questions JSON.")
    st.error("No valid questions found.")
    return []

# Evaluate an answer
def evaluate_answer(question, answer, resume, model):
    prompt = f"""
Evaluate this answer to the question below strictly.
Question: {question['text']}
Answer: {answer}
Candidate resume:
{resume}
Return JSON with fields: score (0-10), justification, 3 improvements, model answer.
"""
    messages = [
        {"role": "system", "content": "You evaluate interview answers."},
        {"role": "user", "content": prompt}
    ]
    resp = call_openai_chat(messages, model=model, max_tokens=700)
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

# Summarize session
def summarize_session(questions, evals, resume, model):
    items = []
    for i, q in enumerate(questions):
        ev = evals[i] if i < len(evals) else {}
        items.append({
            "question": q["text"],
            "score": ev.get("score", 0),
            "justification": ev.get("justification", ""),
            "improvements": ev.get("improvements", []),
            "answer": ev.get("answer", ""),
            "model_answer": ev.get("model_answer", "")
        })
    prompt = f"""
Summarize these interview results strictly.
{json.dumps(items, indent=2)}
Candidate resume:
{resume}
Return JSON: overall score (0-10), strengths [], weaknesses [], recommendation (Yes/No/Maybe), next steps [].
"""
    messages = [
        {"role": "system", "content": "You summarize interview sessions."},
        {"role": "user", "content": prompt}
    ]
    resp = call_openai_chat(messages, model=model, max_tokens=600)
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

# Generate PDF report
def generate_pdf(name, role, exp, questions, evals, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=10)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Interview Report", 0, 1, "C")

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Candidate Name: {name}", 0, 1)
    pdf.cell(0, 10, f"Role: {role}", 0, 1)
    pdf.cell(0, 10, f"Experience: {exp}", 0, 1)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Interview Questions and Answers:", 0, 1)

    pdf.set_font("Arial", size=11)
    for i, q in enumerate(questions):
        pdf.multi_cell(0, 10, f"Q{i+1}: {q['text']}")
        ev = evals[i] if i < len(evals) else {}
        pdf.multi_cell(0, 10, f"Answer: {ev.get('answer', '')}")
        pdf.multi_cell(0, 10, f"Score: {ev.get('score', 0)}")
        pdf.multi_cell(0, 10, f"Justification: {ev.get('justification', '')}")
        pdf.multi_cell(0, 10, f"Improvements: {'; '.join(ev.get('improvements', []))}")
        pdf.multi_cell(0, 10, f"Model Answer: {ev.get('model_answer', '')}")
        pdf.cell(0, 10, "", 0, 1)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary:", 0, 1)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Overall Score: {summary.get('overall_score', '-')} / 10", 0, 1)
    pdf.multi_cell(0, 10, f"Strengths: {'; '.join(summary.get('strengths', []))}")
    pdf.multi_cell(0, 10, f"Weaknesses: {'; '.join(summary.get('weaknesses', []))}")
    pdf.multi_cell(0, 10, f"Recommendation: {summary.get('recommendation', '')}")
    pdf.multi_cell(0, 10, f"Next Steps: {'; '.join(summary.get('next_steps', []))}")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# Assemble session JSON
def make_session_json(name, role, exp, resume, questions, evals):
    summary = summarize_session(questions, evals, resume, st.session_state.get("model", "gpt-4o"))
    return {
        "candidate": {"name": name, "role": role, "experience": exp, "resume": resume},
        "interview": {
            "questions": questions,
            "evaluations": evals,
            "summary": summary,
            "date": datetime.now().isoformat()
        }
    }

# === Streamlit UI Components ===

def sidebar_ui():
    st.sidebar.header("Settings & API Key")
    key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here")
    st.session_state["openai_api_key"] = key
    st.sidebar.markdown("---")

def resume_input_ui():
    st.header("Step 1: Upload Resume or Use Demo")
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    use_demo = st.checkbox("Use Demo Resume")
    if uploaded_file:
        try:
            txt = extract_text(uploaded_file)
            st.success("Resume loaded successfully.")
            st.text_area("Resume Preview", txt, height=200)
            return txt
        except Exception as e:
            st.error(f"Error reading resume: {e}")
            return None
    elif use_demo:
        st.info("Using demo resume.")
        st.text_area("Resume Preview", DEFAULT_DEMO, height=200)
        return DEFAULT_DEMO
    else:
        st.warning("Please upload a resume file or select demo resume to continue.")
        return None

def candidate_info_ui():
    st.header("Step 2: Candidate Information")
    name = st.text_input("Candidate Name")
    role = st.text_input("Role / Position", value="Software Engineer")
    exp = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager"])
    num_qs = st.slider("Number of Interview Questions", min_value=3, max_value=10, value=5)
    model_choice = st.selectbox("Select OpenAI Model", options=list(MODELS.keys()), index=0)
    return name, role, exp, num_qs, MODELS[model_choice]

def interview_ui(questions, resume, model):
    st.header("Step 3: Interview")
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.evaluations = []

    idx = st.session_state.current_question

    if idx >= len(questions):
        st.success("You have completed the interview.")
        return st.session_state.evaluations

    question = questions[idx]
    st.subheader(f"Question {idx + 1} [{question.get('topic', '')} | {question.get('difficulty', '')}]")
    st.write(question["text"])

    # Text answer (no audio for simplicity)
    answer = st.text_area(f"Your Answer for question {idx + 1}", key=f"answer_{idx}", height=150)

    if st.button("Submit Answer", key=f"submit_{idx}"):
        if not answer.strip():
            st.warning("Please provide your answer before submitting.")
            st.stop()
        with st.spinner("Evaluating your answer..."):
            eval_result = evaluate_answer(question, answer.strip(), resume, model)
            eval_result["answer"] = answer.strip()
        st.session_state.evaluations.append(eval_result)
        st.session_state.current_question += 1
        st.experimental_rerun()

    st.info("Submit your answer by clicking the button above.")

def summary_ui(name, role, exp, resume, questions, evaluations):
    st.header("Step 4: Summary & Export")

    summary = summarize_session(questions, evaluations, resume, st.session_state.get("model", "gpt-4o"))

    st.subheader(f"Overall Score: {summary.get('overall_score', '-')}/10")
    st.markdown("**Strengths:**")
    for s in summary.get("strengths", []):
        st.write(f"- {s}")

    st.markdown("**Weaknesses:**")
    for w in summary.get("weaknesses", []):
        st.write(f"- {w}")

    st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")

    st.markdown("**Next Steps:**")
    for step in summary.get("next_steps", []):
        st.write(f"- {step}")

    pdf_buffer = generate_pdf(name, role, exp, questions, evaluations, summary)
    st.download_button(
        "Download PDF Report",
        pdf_buffer,
        file_name=f"{name.replace(' ', '_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

    session_json = make_session_json(name, role, exp, resume, questions, evaluations)
    session_bytes = json.dumps(session_json, indent=2).encode()
    st.download_button(
        "Download Session JSON",
        session_bytes,
        file_name=f"{name.replace(' ', '_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

    if st.button("Save Session"):
        os.makedirs(SESSION_DIR, exist_ok=True)
        with open(
            os.path.join(
                SESSION_DIR, f"{name.replace(' ', '_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ),
            "w",
        ) as f:
            json.dump(session_json, f, indent=2)
        st.success("Session saved successfully! Please restart the app for a new session.")
        # Clear state to start fresh
        for key in ["current_question", "evaluations", "questions", "resume", "candidate_name", "role", "experience"]:
            if key in st.session_state:
                del st.session_state[key]
        st.stop()

# === Main ===

def main():
    st.set_page_config(page_title="AI Interviewer", layout="wide")
    sidebar_ui()

    # Step 1: Resume Input
    if "resume" not in st.session_state:
        resume_text = resume_input_ui()
        if not resume_text:
            st.stop()
        st.session_state.resume = resume_text

    # Step 2: Candidate Info
    if "candidate_name" not in st.session_state:
        name, role, exp, num_questions, model = candidate_info_ui()
        if not name.strip():
            st.warning("Please enter the candidate name.")
            st.stop()
        st.session_state.candidate_name = name.strip()
        st.session_state.role = role.strip()
        st.session_state.experience = exp
        st.session_state.num_questions = num_questions
        st.session_state.model = model

    # Step 3: Generate questions
    if "questions" not in st.session_state:
        with st.spinner("Generating questions..."):
            qs = gen_questions(
                st.session_state.resume,
                st.session_state.role,
                st.session_state.experience,
                st.session_state.num_questions,
                st.session_state.model,
            )
        if not qs:
            st.error("Failed to generate interview questions.")
            st.stop()
        st.session_state.questions = qs

    # Step 4: Conduct Interview
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []

    evals = interview_ui(st.session_state.questions, st.session_state.resume, st.session_state.model)
    if evals and len(evals) == len(st.session_state.questions):
        summary_ui(
            st.session_state.candidate_name,
            st.session_state.role,
            st.session_state.experience,
            st.session_state.resume,
            st.session_state.questions,
            evals,
        )
        
if __name__ == "__main__":
    main()
