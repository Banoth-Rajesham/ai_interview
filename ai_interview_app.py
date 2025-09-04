import streamlit as st
import openai
import PyPDF2
import io
import json
import os
from fpdf import FPDF
from datetime import datetime
import re
import av
import base64
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

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
        
def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)


# --- Core Logic Functions (Implemented) ---

def generate_questions(resume, role, experience, num_questions, model):
    prompt = f"""
    Based on the following resume and job description, generate {num_questions} interview questions.
    **Resume:** {resume}
    **Role:** {role}
    **Experience Level:** {experience}
    Instructions: Create a mix of technical, behavioral, and project-based questions. For each question, specify a 'topic' (e.g., "Python", "System Design", "Behavioral") and a 'difficulty' ("Easy", "Medium", "Hard"). Return a JSON-formatted list of dictionaries with keys: "text", "topic", "difficulty".
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.5, max_tokens=1500)
        content = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        st.error("Failed to parse questions from AI response.")
        return None
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return None

def evaluate_answer(question, answer, resume, model):
    prompt = f"""
    Evaluate a candidate's answer to an interview question based on their resume.
    **Resume:** {resume}
    **Question:** {question['text']} (Topic: {question['topic']}, Difficulty: {question['difficulty']})
    **Candidate's Answer:** {answer}
    Instructions: Assess technical accuracy, clarity, and depth. Provide a 'score' (1-10), constructive 'feedback', and a 'better_answer'. Return a single JSON object with keys: "score", "feedback", "better_answer".
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.2, max_tokens=1000)
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"score": 0, "feedback": "Error parsing response.", "better_answer": "N/A"}
    except Exception as e:
        return {"score": 0, "feedback": str(e), "better_answer": "N/A"}

def summarize_session(questions, answers, resume, model):
    session_details = "\n".join(
        f"Q: {q['text']}\nA: {a['answer']}\nScore: {a['score']}/10\n---"
        for q, a in zip(questions, answers)
    )
    prompt = f"""
    Provide a final summary of an interview based on the transcript and resume.
    **Resume:** {resume}
    **Full Interview Transcript:** {session_details}
    Instructions: Calculate an 'overall_score' (1-10). List 'strengths' and 'weaknesses'. Provide a final 'recommendation' ("Strong Hire", "Hire", "No Hire"). Suggest 'next_steps'. Return a JSON object with keys: "overall_score", "strengths", "weaknesses", "recommendation", "next_steps". 'strengths', 'weaknesses', 'next_steps' should be lists of strings.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages, model=model, temperature=0.4, max_tokens=1500)
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {}
    except Exception as e:
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
    def write_text(text):
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
    pdf.set_font('Arial', 'B', 16); write_text(f"Candidate: {name}")
    pdf.set_font('Arial', '', 12); write_text(f"Role: {role}\nExperience: {exp}\nDate: {datetime.now().strftime('%Y-%m-%d')}"); pdf.ln(10)
    pdf.set_font('Arial', 'B', 14); write_text("Interview Summary")
    pdf.set_font('Arial', '', 12); write_text(f"Overall Score: {summary.get('overall_score', 'N/A')}/10\nRecommendation: {summary.get('recommendation', 'N/A')}"); pdf.ln(5)
    pdf.set_font('Arial', 'B', 12); write_text("Strengths:"); pdf.set_font('Arial', '', 12)
    for s in summary.get("strengths", []): write_text(f"- {s}")
    pdf.ln(5); pdf.set_font('Arial', 'B', 12); write_text("Weaknesses:"); pdf.set_font('Arial', '', 12)
    for w in summary.get("weaknesses", []): write_text(f"- {w}")
    pdf.ln(10); pdf.set_font('Arial', 'B', 14); write_text("Detailed Question & Answer Analysis")
    for i, (q, a) in enumerate(zip(questions, answers)):
        pdf.set_font('Arial', 'B', 12); write_text(f"Q{i+1}: {q['text']} ({q['topic']} | {q['difficulty']})")
        pdf.set_font('Arial', '', 12); write_text(f"Answer: {a['answer']}")
        pdf.set_font('Arial', 'I', 12); write_text(f"Feedback: {a['feedback']} (Score: {a['score']}/10)"); pdf.ln(10)
    return pdf.output(dest='S').encode('latin-1')

def make_session_json(name, role, exp, resume, questions, answers):
    return {"candidate_name": name, "role": role, "experience": exp, "resume": resume, "questions": questions, "answers": answers, "timestamp": datetime.now().isoformat()}

# --- UI Helpers ---
def sidebar():
    st.sidebar.markdown("# AI Interviewer Settings")
    key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Paste your OpenAI API key here")
    st.session_state["openai_api_key"] = key

def resume_section():
    st.markdown("## Step 1: Upload Resume or Use Demo")
    uploaded_file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])
    use_demo = st.checkbox("Use Demo Resume")
    resume = None
    if uploaded_file:
        resume = extract_text(uploaded_file)
        if resume: st.text_area("Resume Preview", resume, height=150)
    elif use_demo:
        st.text_area("Demo Resume Preview", DEFAULT_DEMO, height=150)
        resume = DEFAULT_DEMO
    return resume

def candidate_section():
    st.markdown("## Step 2: Candidate Information")
    name = st.text_input("Candidate Name")
    role = st.text_input("Position / Role", "Software Engineer")
    exp = st.selectbox("Experience Level", ["Intern", "Junior", "Mid-Level", "Senior"])
    q_count = st.slider("Number of Questions", 3, 10, 5)
    model_key = st.selectbox("OpenAI Model", list(MODELS.keys()), index=0)
    return name.strip(), role.strip(), exp, q_count, MODELS[model_key]

def interview_section(questions, resume, model):
    st.markdown("## Step 3: Interview")
    idx = st.session_state.current_index
    if idx >= len(questions):
        st.success("Interview completed! Proceeding to summary.")
        return

    q = questions[idx]
    st.subheader(f"Question {idx+1}: {q.get('topic','')} ({q.get('difficulty','')})")
    st.write(q["text"])

    if f"tts_{idx}" not in st.session_state:
        with st.spinner("Generating question audio..."):
            audio_resp = text_to_speech(q["text"])
            if audio_resp: st.session_state[f"tts_{idx}"] = audio_resp.content
    if st.session_state.get(f"tts_{idx}"):
        autoplay_audio(st.session_state[f"tts_{idx}"])

    st.markdown("### Record your answer")
    
    # This is where we will store the recorded audio frames
    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = []

    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.audio_buffer = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # We must return the frame to avoid breaking the stream
            self.audio_buffer.append(frame.to_ndarray().tobytes())
            return frame

    webrtc_ctx = webrtc_streamer(
        key=f"audio_recorder_{idx}",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        st.session_state.audio_buffer.extend(webrtc_ctx.audio_processor.audio_buffer)
        webrtc_ctx.audio_processor.audio_buffer.clear()
        
    if st.button("Stop and Submit Answer", key=f"submit_{idx}"):
        if not st.session_state.audio_buffer:
            st.warning("Please record an answer before submitting.")
            return

        # Combine audio chunks. 
        # Note: This is a simplified approach. For production, you'd want to properly format it as a WAV file.
        full_audio_bytes = b"".join(st.session_state.audio_buffer)
        
        # Clear buffer for next question
        st.session_state.audio_buffer = []

        with st.spinner("Transcribing and evaluating your answer..."):
            answer_text = transcribe_audio(full_audio_bytes)
            if answer_text:
                st.info(f"**Your Answer (Transcribed):** {answer_text}")
                evaluation = evaluate_answer(q, answer_text, resume, model)
                evaluation["answer"] = answer_text
                st.session_state.answers.append(evaluation)
                st.session_state.current_index += 1
                st.rerun()
            else:
                st.error("Transcription failed. Please try recording your answer again.")

def summary_section(name, role, exp, resume, questions, answers):
    st.markdown("## Step 4: Summary & Export")
    with st.spinner("Generating overall summary..."):
        summary = summarize_session(questions, answers, resume, st.session_state.model)

    st.subheader(f"Overall Score: {summary.get('overall_score', '-')}/10")
    st.markdown(f"**Recommendation:** {summary.get('recommendation', '')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths:**")
        for strength in summary.get("strengths", []): st.write(f"- {strength}")
    with col2:
        st.markdown("**Weaknesses:**")
        for weakness in summary.get("weaknesses", []): st.write(f"- {weakness}")
    pdf_buffer = generate_pdf(name, role, exp, questions, answers, summary)
    st.download_button("Download PDF Report", pdf_buffer, f"{name.replace(' ','_')}_Report.pdf", "application/pdf")

def main():
    st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")
    st.title("🧠 AI Interviewer")
    sidebar()
    
    # Initialize state
    if "current_index" not in st.session_state: st.session_state.current_index = 0
    if "answers" not in st.session_state: st.session_state.answers = []

    # State Machine
    if "questions" not in st.session_state:
        resume = resume_section()
        if resume:
            st.session_state.resume = resume
            name, role, exp, q_count, model = candidate_section()
            if name and st.button("Start Interview"):
                st.session_state.update({"candidate_name": name, "role": role, "experience": exp, "num_questions": q_count, "model": model})
                with st.spinner("Generating questions..."):
                    questions = generate_questions(st.session_state.resume, role, exp, q_count, model)
                    if questions:
                        st.session_state.questions = questions
                        st.rerun()
                    else:
                        st.error("Failed to generate questions.")
    elif st.session_state.current_index < len(st.session_state.questions):
        interview_section(st.session_state.questions, st.session_state.resume, st.session_state.model)
    else:
        summary_section(st.session_state.candidate_name, st.session_state.role, st.session_state.experience, st.session_state.resume, st.session_state.questions, st.session_state.answers)

if __name__ == "__main__":
    main()```
