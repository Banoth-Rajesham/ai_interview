import streamlit as st
import openai
import PyPDF2
import io
import json
import os
from fpdf import FPDF
from datetime import datetime
import re
import av # Required for streamlit-webrtc

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

# --- Utils ---
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
            # When passing bytes, you need to give it a name
            file.name = "interview_answer.wav"
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

# --- Core Logic Functions (Previously Missing) ---

def generate_questions(resume, role, experience, num_questions, model):
    prompt = f"""
    Based on the following resume and job description, generate {num_questions} interview questions.

    **Resume:**
    {resume}

    **Role:** {role}
    **Experience Level:** {experience}

    Instructions:
    1. Create a mix of questions covering technical skills, behavioral aspects, and project experiences relevant to the resume and role.
    2. For each question, specify a 'topic' (e.g., "Python", "System Design", "Behavioral") and a 'difficulty' (e.g., "Easy", "Medium", "Hard").
    3. Return the output as a JSON-formatted string representing a list of dictionaries. Each dictionary must have three keys: "text", "topic", and "difficulty".

    Example format:
    [
        {{"text": "Explain the difference between a list and a tuple in Python.", "topic": "Python", "difficulty": "Easy"}},
        {{"text": "Describe a challenging project you worked on and how you handled it.", "topic": "Behavioral", "difficulty": "Medium"}}
    ]
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.5, max_tokens=1500)
        content = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        st.error("Failed to parse questions from the AI response.")
        return None
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return None

def evaluate_answer(question, answer, resume, model):
    prompt = f"""
    Evaluate a candidate's answer to an interview question based on their resume.

    **Resume:**
    {resume}

    **Question:**
    {question['text']}
    (Topic: {question['topic']}, Difficulty: {question['difficulty']})

    **Candidate's Answer:**
    {answer}

    Instructions:
    1. Assess the answer's technical accuracy, clarity, and depth.
    2. Provide a 'score' from 1 to 10.
    3. Give constructive 'feedback' on what was good and what could be improved.
    4. Provide a 'better_answer' that exemplifies a strong response.
    5. Return the output as a single JSON object with keys: "score", "feedback", and "better_answer".
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.2, max_tokens=1000)
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        st.error("Failed to parse evaluation from the AI response.")
        return {"score": 0, "feedback": "Error parsing response.", "better_answer": "N/A"}
    except Exception as e:
        st.error(f"Error evaluating answer: {e}")
        return {"score": 0, "feedback": str(e), "better_answer": "N/A"}

def summarize_session(questions, answers, resume, model):
    session_details = "\n".join(
        f"Q: {q['text']}\nA: {a['answer']}\nScore: {a['score']}/10\nFeedback: {a['feedback']}\n---"
        for q, a in zip(questions, answers)
    )
    prompt = f"""
    Based on the entire interview transcript and the candidate's resume, provide a final summary.

    **Resume:**
    {resume}

    **Full Interview Transcript:**
    {session_details}

    Instructions:
    1. Calculate an 'overall_score' out of 10.
    2. List the key 'strengths' demonstrated by the candidate.
    3. List the key 'weaknesses' or areas for improvement.
    4. Provide a final 'recommendation' (e.g., "Strong Hire", "Hire with reservations", "No Hire").
    5. Suggest 'next_steps' for the candidate (e.g., "Proceed to next round", "Additional technical screening").
    6. Return a single JSON object with keys: "overall_score", "strengths", "weaknesses", "recommendation", "next_steps". 'strengths', 'weaknesses', and 'next_steps' should be lists of strings.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.4, max_tokens=1500)
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        st.error("Failed to parse summary from the AI response.")
        return {}
    except Exception as e:
        st.error(f"Error summarizing session: {e}")
        return {}

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Interview Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(name, role, exp, questions, answers, summary):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # Helper to write text and handle unicode
    def write_text(text):
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))

    # Basic Info
    pdf.set_font('Arial', 'B', 16)
    write_text(f"Candidate: {name}")
    pdf.set_font('Arial', '', 12)
    write_text(f"Role: {role}")
    write_text(f"Experience: {exp}")
    write_text(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    pdf.ln(10)

    # Summary
    pdf.set_font('Arial', 'B', 14)
    write_text("Interview Summary")
    pdf.set_font('Arial', '', 12)
    write_text(f"Overall Score: {summary.get('overall_score', 'N/A')}/10")
    write_text(f"Recommendation: {summary.get('recommendation', 'N/A')}")
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    write_text("Strengths:")
    pdf.set_font('Arial', '', 12)
    for s in summary.get("strengths", []):
        write_text(f"- {s}")
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    write_text("Weaknesses:")
    pdf.set_font('Arial', '', 12)
    for w in summary.get("weaknesses", []):
        write_text(f"- {w}")
    pdf.ln(10)

    # Detailed Q&A
    pdf.set_font('Arial', 'B', 14)
    write_text("Detailed Question & Answer Analysis")
    for i, (q, a) in enumerate(zip(questions, answers)):
        pdf.set_font('Arial', 'B', 12)
        write_text(f"Question {i+1}: {q['text']} ({q['topic']} | {q['difficulty']})")
        pdf.set_font('Arial', '', 12)
        write_text(f"Answer: {a['answer']}")
        pdf.set_font('Arial', 'I', 12)
        write_text(f"Feedback: {a['feedback']} (Score: {a['score']}/10)")
        pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')

def make_session_json(name, role, exp, resume, questions, answers):
    return {
        "candidate_name": name,
        "role": role,
        "experience": exp,
        "resume": resume,
        "questions": questions,
        "answers": answers,
        "timestamp": datetime.now().isoformat()
    }

# --- UI Helpers ---
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
    resume = None
    if uploaded_file:
        resume = extract_text(uploaded_file)
        if resume:
            st.success("Resume uploaded and parsed successfully!")
            st.text_area("Resume Preview", resume, height=200)
    elif use_demo:
        st.info("Using demo resume")
        st.text_area("Demo Resume Preview", DEFAULT_DEMO, height=200)
        resume = DEFAULT_DEMO
    else:
        st.warning("Please upload a resume or select demo to proceed.")
    return resume

def candidate_section():
    st.markdown("## Step 2: Candidate Information")
    name = st.text_input("Candidate Name", "")
    role = st.text_input("Position / Role", "Software Engineer")
    exp = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior", "Lead", "Manager"])
    question_count = st.slider("Number of Interview Questions", 3, 10, 5)
    model_key = st.selectbox("Choose OpenAI Model", list(MODELS.keys()), index=0)
    if not name.strip():
        st.warning("Please enter the candidate's name to continue.")
        return None, None, None, None, None
    return name.strip(), role.strip(), exp, question_count, MODELS[model_key]

def interview_section(questions, resume, model):
    st.markdown("## Step 3: Interview")
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    idx = st.session_state["current_index"]

    if idx >= len(questions):
        st.success("Interview completed! Proceeding to summary.")
        return st.session_state.get("answers", [])

    q = questions[idx]
    st.subheader(f"Question {idx+1} ({q.get('topic','')} | {q.get('difficulty','')})")
    st.write(q["text"])

    if f"tts_{idx}" not in st.session_state:
        with st.spinner("Generating audio for the question..."):
            audio_resp = text_to_speech(q["text"])
            if audio_resp:
                st.session_state[f"tts_{idx}"] = audio_resp.content
    if st.session_state.get(f"tts_{idx}"):
        st.audio(st.session_state[f"tts_{idx}"], format="audio/mp3")

    st.markdown("### Record your answer or type below:")
    webrtc_ctx = webrtc_streamer(
        key=f"audio_recorder_{idx}",
        mode=st.webrtc.WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
    )

    audio_buffer = io.BytesIO()
    if webrtc_ctx.audio_receiver:
        try:
            sound_chunk = webrtc_ctx.audio_receiver.get_frame(timeout=1)
            # This part is complex. For a simple recorder, you'd collect frames.
            # A simpler approach is to handle recording outside the main script flow.
            # For now, we will rely on a typed answer primarily.
        except Exception:
            pass

    answer_text = st.text_area("Your Answer:", key=f"typed_answer_{idx}", height=150)

    if st.button("Submit Answer", key=f"submit_{idx}"):
        if not answer_text.strip():
            st.warning("Please provide an answer before submitting.")
        else:
            with st.spinner("Evaluating answer..."):
                evaluation = evaluate_answer(q, answer_text, resume, model)
                evaluation["answer"] = answer_text
            if "answers" not in st.session_state:
                st.session_state["answers"] = []
            st.session_state["answers"].append(evaluation)
            st.session_state["current_index"] += 1
            st.rerun()

    st.info("Press 'Submit Answer' when you are ready.")
    return None

def summary_section(name, role, exp, resume, questions, answers):
    st.markdown("## Step 4: Summary & Export")
    with st.spinner("Generating overall summary..."):
        summary = summarize_session(questions, answers, resume, st.session_state.get("model", "gpt-4o"))

    st.subheader(f"Overall Score: {summary.get('overall_score', '-')}/10")
    st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths:**")
        for strength in summary.get("strengths", []):
            st.write(f"- {strength}")
    with col2:
        st.markdown("**Weaknesses:**")
        for weakness in summary.get("weaknesses", []):
            st.write(f"- {weakness}")

    st.markdown("**Next Steps:**")
    for step in summary.get("next_steps", []):
        st.write(f"- {step}")

    pdf_buffer = generate_pdf(name, role, exp, questions, answers, summary)
    st.download_button(
        "Download PDF Report",
        pdf_buffer,
        file_name=f"{name.replace(' ','_')}_Interview_Report.pdf",
        mime="application/pdf",
    )

    session_json = make_session_json(name, role, exp, resume, questions, answers)
    json_bytes = json.dumps(session_json, indent=2).encode()
    st.download_button(
        "Download Session JSON",
        json_bytes,
        file_name=f"{name.replace(' ','_')}_Interview_Session.json",
        mime="application/json",
    )

    if st.button("Save Session & Start New"):
        filename = os.path.join(SESSION_DIR, f"{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(filename, "w") as f:
            json.dump(session_json, f, indent=2)
        st.success(f"Session saved to {filename}!")
        
        # Clear state to restart
        keys_to_clear = ["resume", "candidate_name", "role", "experience", "num_questions", "model", "questions", "answers", "current_index"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

def main():
    st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")
    st.title("🧠 AI Interviewer")
    sidebar()

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "answers" not in st.session_state:
        st.session_state.answers = []

    # State Machine Logic
    if "questions" not in st.session_state:
        # State 1: Collect Resume and Candidate Info
        resume = resume_section()
        if resume:
            st.session_state.resume = resume
            name, role, exp, q_count, model = candidate_section()
            if name:
                st.session_state.candidate_name = name
                st.session_state.role = role
                st.session_state.experience = exp
                st.session_state.num_questions = q_count
                st.session_state.model = model
                if st.button("Start Interview"):
                    with st.spinner("Generating interview questions..."):
                        questions = generate_questions(st.session_state.resume, role, exp, q_count, model)
                        if questions:
                            st.session_state.questions = questions
                            st.rerun()
                        else:
                            st.error("Failed to generate questions. Check API key and settings.")
    elif st.session_state.current_index < len(st.session_state.questions):
        # State 2: Conduct Interview
        interview_section(
            st.session_state.questions,
            st.session_state.resume,
            st.session_state.model
        )
    else:
        # State 3: Show Summary
        summary_section(
            st.session_state.candidate_name,
            st.session_state.role,
            st.session_state.experience,
            st.session_state.resume,
            st.session_state.questions,
            st.session_state.answers
        )

if __name__ == "__main__":
    main()
