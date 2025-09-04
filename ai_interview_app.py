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

# --- Constants and Config ---
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

# --- Helper Functions ---

def get_openai_key():
    # 1. Try session state (sidebar input)
    key = st.session_state.get("openai_api_key", "")
    if key:
        return key
    # 2. Try Streamlit Secrets (recommended secure method)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # 3. Try environment variables
    return os.getenv("OPENAI_API_KEY", "")

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return clean_text(text)
    elif uploaded_file.name.lower().endswith(".txt"):
        return clean_text(uploaded_file.read().decode())
    else:
        st.error("Unsupported file type. Upload PDF or TXT.")
        return ""

def clean_text(text):
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def openai_client():
    key = get_openai_key()
    if not key:
        st.error("OpenAI API key not found! Please add it in sidebar or in streamlit secrets.")
        st.stop()
    return openai.OpenAI(api_key=key)

def openai_chat(messages, model="gpt-4o", temperature=0.3, max_tokens=1200):
    client = openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

def generate_questions(resume, role, exp, count, model):
    prompt = f"""
You are an expert interviewer. Based on this resume and the role '{role}' with experience '{exp}', generate exactly {count} interview questions as a JSON array.
Include at least two personalized questions based on the resume. Fields: id, text, topic, difficulty (Easy/Medium/Hard), estimated_time_seconds.
Resume:
{resume}
"""
    messages = [
        {"role": "system", "content": "You generate interview questions."},
        {"role": "user", "content": prompt}
    ]
    resp = openai_chat(messages, model=model, max_tokens=2000)
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
    st.error("Could not generate interview questions.")
    return []

def evaluate_answer(question, answer, resume, model):
    prompt = f"""
You are an interviewer evaluating this answer:
Question: {question['text']}
Answer: {answer}
Resume:
{resume}
Provide JSON: score (0-10), justification, improvements [3], model answer.
"""
    messages = [
        {"role": "system", "content": "You evaluate interview answers."},
        {"role": "user", "content": prompt}
    ]
    resp = openai_chat(messages, model=model, max_tokens=700)
    raw = resp.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            evaluation = json.loads(match.group())
            evaluation["score"] = int(evaluation.get("score", 0))
            evaluation["improvements"] = evaluation.get("improvements", [])[:3]
            return evaluation
        except:
            try:
                evaluation = ast.literal_eval(match.group())
                evaluation["score"] = int(evaluation.get("score", 0))
                evaluation["improvements"] = evaluation.get("improvements", [])[:3]
                return evaluation
            except:
                return {"score": 0, "justification": "Parse failure", "improvements": [], "model_answer": ""}
    return {"score": 0, "justification": "Evaluation failed", "improvements": [], "model_answer": ""}

def summarize_session(questions, evals, resume, model):
    results = []
    for i, q in enumerate(questions):
        ev = evals[i] if i < len(evals) else {}
        results.append({
            "question": q["text"],
            "score": ev.get("score", 0),
            "justification": ev.get("justification", ""),
            "improvements": ev.get("improvements", []),
            "answer": ev.get("answer", ""),
            "model_answer": ev.get("model_answer", "")
        })
    prompt = f"""
Summarize the interview results:
{json.dumps(results, indent=2)}
Candidate resume:
{resume}
Return JSON with: overall_score (0-10 int), strengths ([]), weaknesses ([]), recommendation ("Yes"/"No"/"Maybe"), next_steps ([]).
"""
    messages = [
        {"role": "system", "content": "You summarize interview sessions."},
        {"role": "user", "content": prompt}
    ]
    resp = openai_chat(messages, model=model, max_tokens=600)
    raw = resp.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            summary = json.loads(match.group())
            summary["overall_score"] = int(summary.get("overall_score", 5))
            return summary
        except:
            return {"overall_score": 5, "strengths": [], "weaknesses": [], "recommendation": "Maybe", "next_steps": []}
    return {"overall_score": 5, "strengths": [], "weaknesses": [], "recommendation": "Maybe", "next_steps": []}

def generate_pdf(name, role, exp, questions, evals, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=10)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Interview Report", 0, 1, "C")

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Candidate: {name}", 0, 1)
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
    pdf.cell(0, 10, f"Overall Score: {summary.get('overall_score', '-')}/10", 0, 1)
    pdf.multi_cell(0, 10, f"Strengths: {'; '.join(summary.get('strengths', []))}")
    pdf.multi_cell(0, 10, f"Weaknesses: {'; '.join(summary.get('weaknesses', []))}")
    pdf.multi_cell(0, 10, f"Recommendation: {summary.get('recommendation', '')}")
    pdf.multi_cell(0, 10, f"Next Steps: {'; '.join(summary.get('next_steps', []))}")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

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

# --- UI ---

def sidebar_ui():
    st.sidebar.header("Settings & API Key")
    key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    st.session_state["openai_api_key"] = key
    st.sidebar.markdown("---")

def resume_upload_ui():
    st.header("Step 1: Upload Resume or Use Demo")
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    use_demo = st.checkbox("Use Demo Resume")
    if uploaded_file:
        try:
            txt = extract_text_from_file(uploaded_file)
            st.success("Resume loaded successfully.")
            st.text_area("Resume Preview", txt, height=200)
            return txt
        except Exception as e:
            st.error(f"Error loading resume: {e}")
            return None
    elif use_demo:
        st.info("Using demo resume")
        st.text_area("Resume Preview", DEFAULT_DEMO, height=200)
        return DEFAULT_DEMO
    else:
        st.warning("Please upload a resume or select demo resume.")
        return None

def candidate_info_ui():
    st.header("Step 2: Candidate Information")
    name = st.text_input("Candidate Name")
    role = st.text_input("Role / Position", value="Software Engineer")
    exp = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager"])
    num_qs = st.slider("Number of Interview Questions", 3, 10, 5)
    model_choice = st.selectbox("OpenAI Model", list(MODELS.keys()), index=0)
    return name, role, exp, num_qs, MODELS[model_choice]

def interview_ui(questions, resume, model):
    st.header("Step 3: Interview")
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = 0
        st.session_state["evaluations"] = []

    idx = st.session_state["current_question"]

    if idx >= len(questions):
        st.success("Interview complete!")
        return st.session_state["evaluations"]

    question = questions[idx]
    st.subheader(f"Question {idx+1} [{question.get('topic','')} | {question.get('difficulty','')}]")
    st.write(question["text"])

    answer = st.text_area("Your answer", key=f"answer_{idx}", height=150)

    if st.button("Submit Answer", key=f"submit_{idx}"):
        if not answer.strip():
            st.warning("Please provide your answer before submitting.")
            st.stop()
        with st.spinner("Evaluating answer..."):
            ev = evaluate_answer(question, answer.strip(), resume, model)
            ev["answer"] = answer.strip()
        st.session_state["evaluations"].append(ev)
        st.session_state["current_question"] += 1
        st.experimental_rerun()

    st.info("Submit your answer by clicking the button above.")

def summary_ui(name, role, exp, resume, questions, evaluations):
    st.header("Step 4: Summary and Export")
    summary = summarize_session(questions, evaluations, resume, st.session_state.get("model", "gpt-4o"))

    st.subheader(f"Overall Score: {summary.get('overall_score','-')} / 10")

    st.markdown("**Strengths:**")
    for s in summary.get("strengths", []):
        st.write(f"- {s}")

    st.markdown("**Weaknesses:**")
    for w in summary.get("weaknesses", []):
        st.write(f"- {w}")

    st.markdown(f"**Recommendation:** {summary.get('recommendation','')}")

    st.markdown("**Next Steps:**")
    for ns in summary.get("next_steps", []):
        st.write(f"- {ns}")

    pdf_buf = generate_pdf(name, role, exp, questions, evaluations, summary)
    st.download_button(
        "Download PDF Report",
        pdf_buf,
        file_name=f"{name.replace(' ','_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

    session_json = make_session_json(name, role, exp, resume, questions, evaluations)
    json_bytes = json.dumps(session_json, indent=2).encode()
    st.download_button(
        "Download Session JSON",
        json_bytes,
        file_name=f"{name.replace(' ','_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

    if st.button("Save Session"):
        os.makedirs(SESSION_DIR, exist_ok=True)
        filepath = os.path.join(
            SESSION_DIR, f"{name.replace(' ','_')}_Interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filepath, "w") as f:
            json.dump(session_json, f, indent=2)
        st.success("Session saved successfully! Please restart the app for a new session.")
        # Clear state to start fresh next time
        for key in ["current_question", "evaluations", "questions", "resume", "candidate_name", "role", "experience"]:
            if key in st.session_state:
                del st.session_state[key]
        st.stop()

def main():
    st.set_page_config(page_title="AI Interviewer", layout="wide", page_icon="🤖")
    sidebar_ui()

    if "resume" not in st.session_state:
        resume = resume_upload_ui()
        if not resume:
            st.stop()
        st.session_state["resume"] = resume

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

    if "evaluations" not in st.session_state:
        st.session_state["evaluations"] = []

    evals = interview_ui(st.session_state["questions"], st.session_state["resume"], st.session_state["model"])

    if evals and len(evals) == len(st.session_state["questions"]):
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
