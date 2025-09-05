import streamlit as st
import openai
import PyPDF2
import io
import json
import re
from fpdf import FPDF
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")

MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}

# --- OpenAI helpers ---
def get_openai_key():
    key = st.session_state.get("openai_api_key") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return key

def openai_client():
    return openai.OpenAI(api_key=get_openai_key())

def chat_completion(messages, model="gpt-4o", temperature=0.3, max_tokens=1500):
    client = openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

# --- Resume extraction ---
def extract_text(file):
    try:
        if file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        elif file.type == "text/plain":
            return file.getvalue().decode("utf-8").strip()
        else:
            st.error("Unsupported file type. Please upload PDF or TXT.")
            return None
    except Exception as e:
        st.error(f"Failed to extract text from resume: {e}")
        return None

# --- Question generation ---
def generate_questions(resume, role, num_questions, model):
    prompt = (
        f"Generate {num_questions} interview questions for a {role} based on this resume:\n{resume}\n"
        "Return a JSON list of questions with fields: text, topic, difficulty."
    )
    messages = [{"role": "user", "content": prompt}]
    content = chat_completion(messages, model=model)
    try:
        match = re.search(r"\$.*\$", content, re.DOTALL)
        questions = json.loads(match.group(0)) if match else None
        if not questions:
            st.warning("Could not parse questions from AI response.")
        return questions
    except Exception as e:
        st.error(f"Error parsing questions JSON: {e}")
        return None

# --- Answer evaluation ---
def evaluate_answer(question_text, answer_text, resume, model):
    prompt = (
        f"Given the resume:\n{resume}\n"
        f"Evaluate the answer to the question:\n{question_text}\n"
        f"Answer:\n{answer_text}\n"
        "Return a JSON with fields: score (1-10), feedback, better_answer."
    )
    messages = [{"role": "user", "content": prompt}]
    content = chat_completion(messages, model=model)
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        evaluation = json.loads(match.group(0)) if match else None
        if not evaluation:
            return {"score": 0, "feedback": "No evaluation returned.", "better_answer": ""}
        return evaluation
    except Exception as e:
        st.error(f"Error parsing evaluation JSON: {e}")
        return {"score": 0, "feedback": "Error parsing evaluation.", "better_answer": ""}

# --- Summary generation ---
def summarize_interview(questions, answers, resume, model):
    transcript = ""
    for q, a in zip(questions, answers):
        transcript += f"Q: {q['text']}\nA: {a['answer']}\nScore: {a.get('score', 0)}/10\n\n"
    prompt = (
        f"Summarize the interview based on the resume:\n{resume}\n"
        f"and the transcript:\n{transcript}\n"
        "Return a JSON with overall_score, strengths (list), weaknesses (list), recommendation."
    )
    messages = [{"role": "user", "content": prompt}]
    content = chat_completion(messages, model=model)
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        summary = json.loads(match.group(0)) if match else None
        if not summary:
            st.warning("Could not parse summary from AI response.")
        return summary
    except Exception as e:
        st.error(f"Error parsing summary JSON: {e}")
        return None

# --- PDF generation ---
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AI Interview Report", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_pdf(name, role, summary, questions, answers):
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
        pdf.multi_cell(0, 10, f"Q{i}: {q['text']}")
        pdf.multi_cell(0, 10, f"Answer: {a['answer']}")
        pdf.multi_cell(0, 10, f"Feedback: {a.get('feedback', '')} (Score: {a.get('score', 'N/A')}/10)")
        pdf.ln(5)
    return pdf.output(dest="S").encode("latin-1")

# --- Sidebar ---
def sidebar():
    st.sidebar.title("Settings")
    key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
    if key:
        st.session_state["openai_api_key"] = key

# --- Main app ---
def main():
    sidebar()
    st.title("🧠 AI Interviewer")

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
                            st.experimental_rerun()
                        else:
                            st.error("Failed to generate questions.")
            else:
                st.warning("Could not extract text from resume.")

    elif st.session_state.stage == "interview":
        questions = st.session_state.get("questions", [])
        answers = st.session_state.get("answers", [])
        current_q = st.session_state.get("current_q", 0)

        if current_q >= len(questions):
            st.session_state.stage = "summary"
            st.experimental_rerun()
            return

        q = questions[current_q]
        st.header(f"Question {current_q + 1} of {len(questions)}")
        st.write(f"**Topic:** {q.get('topic', 'General')}")
        st.write(f"**Difficulty:** {q.get('difficulty', 'Medium')}")
        st.write(f"**Question:** {q['text']}")

        answer = st.text_area("Your Answer", key=f"answer_{current_q}")

        if st.button("Submit Answer"):
            if not answer.strip():
                st.warning("Please enter an answer before submitting.")
            else:
                with st.spinner("Evaluating answer..."):
                    evaluation = evaluate_answer(q['text'], answer.strip(), st.session_state.resume, MODELS["GPT-4o"])
                evaluation["answer"] = answer.strip()
                answers.append(evaluation)
                st.session_state.answers = answers
                st.session_state.current_q = current_q + 1
                st.experimental_rerun()

        if st.button("Skip Question"):
            answers.append({"answer": "", "score": 0, "feedback": "Skipped", "better_answer": ""})
            st.session_state.answers = answers
            st.session_state.current_q = current_q + 1
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
            for s in summary.get("strengths", []):
                st.write(f"- {s}")

            st.markdown("**Weaknesses:**")
            for w in summary.get("weaknesses", []):
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
                for key in ["stage", "candidate_name", "role", "resume", "num_questions", "questions", "answers", "current_q"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()
        else:
            st.error("Failed to generate summary.")

if __name__ == "__main__":
    main()
