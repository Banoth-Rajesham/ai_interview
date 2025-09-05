# app.py
# ==============================================================
# 🧠 AI Interviewer with Live Video/Audio — Production-Ready Single File
# ==============================================================

# -----------------------------
# Imports
# -----------------------------
import os
import io
import re
import json
import time
import math
import base64
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from fpdf import FPDF
from openai import OpenAI
from PyPDF2 import PdfReader
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# -----------------------------
# Constants and Configuration
# -----------------------------
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")

DEFAULT_MODEL = "gpt-4o"
SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
TTS_MODEL = "tts-1"
WHISPER_MODEL = "whisper-1"
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

MAX_RESUME_PREVIEW_CHARS = 8000
DEFAULT_NUM_QUESTIONS = 5
DEFAULT_SNAPSHOT_INTERVAL = 10
MAX_TOKENS_COMPLETIONS = 1500
REQUEST_TIMEOUT_SECS = 45

# -----------------------------
# Prompt Templates
# -----------------------------
PROMPT_TEMPLATES: Dict[str, str] = {
    "questions": (
        "You are an expert interviewer. Generate {num} technical and behavioral interview questions "
        "for a candidate applying to the role: \"{role}\" using the resume below. "
        "Vary topic and difficulty. Return STRICT JSON ONLY with this schema:\n"
        "{{\n"
        "  \"questions\": [\n"
        "    {{\"id\": \"q1\", \"text\": \"...\", \"topic\": \"...\", \"difficulty\": \"Easy|Medium|Hard\"}},\n"
        "    ...\n"
        "  ]\n"
        "}}\n\n"
        "Resume:\n{resume}"
    ),
    "evaluate": (
        "You are a senior interviewer. Evaluate the candidate's answer concisely. "
        "Return STRICT JSON ONLY with this schema:\n"
        "{{\"score\": 0-10, \"feedback\": \"short feedback\", \"better_answer\": \"concise improved answer\"}}\n\n"
        "Resume (excerpt):\n{resume}\n\n"
        "Question:\n{question}\n\n"
        "Answer:\n{answer}"
    ),
    "summary": (
        "You are a hiring manager. Summarize the interview succinctly. "
        "Return STRICT JSON ONLY with this schema:\n"
        "{{\"overall_score\": 0-10, \"strengths\": [\"...\"], \"weaknesses\": [\"...\"], \"recommendation\": \"hire or not with rationale\"}}\n\n"
        "Resume (excerpt):\n{resume}\n\n"
        "Transcript:\n{transcript}"
    ),
}

# -----------------------------
# Utilities: JSON Parsing & Sanitization
# -----------------------------
def strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^```
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()

def extract_first_json_block(text: str) -> Optional[Any]:
    if not text:
        return None
    # Fast path if the entire content is JSON
    raw = strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Scan for first valid object or array
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == open_char:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_char and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = raw[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        continue
    return None

def validate_questions(obj: Any) -> Optional[List[Dict[str, Any]]]:
    # Accepts either { "questions": [...] } or a direct list
    if isinstance(obj, dict) and "questions" in obj and isinstance(obj["questions"], list):
        items = obj["questions"]
    elif isinstance(obj, list):
        items = obj
    else:
        return None
    cleaned = []
    for i, q in enumerate(items, start=1):
        if not isinstance(q, dict):
            continue
        qid = q.get("id") or f"q{i}"
        txt = (q.get("text") or "").strip()
        topic = (q.get("topic") or "General").strip()
        diff = (q.get("difficulty") or "Medium").strip()
        if not txt:
            continue
        if diff not in ["Easy", "Medium", "Hard"]:
            diff = "Medium"
        cleaned.append({"id": qid, "text": txt, "topic": topic, "difficulty": diff})
    return cleaned or None

def validate_eval(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    try:
        score = int(obj.get("score", 0))
    except Exception:
        score = 0
    score = max(0, min(10, score))
    feedback = (obj.get("feedback") or "").strip()
    better = (obj.get("better_answer") or "").strip()
    return {"score": score, "feedback": feedback, "better_answer": better}

def validate_summary(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    try:
        overall = int(obj.get("overall_score", 0))
    except Exception:
        overall = 0
    overall = max(0, min(10, overall))
    strengths = obj.get("strengths") or []
    weaknesses = obj.get("weaknesses") or []
    rec = (obj.get("recommendation") or "").strip()
    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    if not isinstance(weaknesses, list):
        weaknesses = [str(weaknesses)]
    return {"overall_score": overall, "strengths": strengths, "weaknesses": weaknesses, "recommendation": rec}

def sanitize_html(s: str) -> str:
    # Minimal sanitizer: strip script tags and on* attributes
    if not s:
        return s
    s = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"on\w+\s*=\s*['\"].*?['\"]", "", s, flags=re.IGNORECASE)
    return s

# -----------------------------
# Resilient OpenAI Client
# -----------------------------
def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Missing OpenAI API key")
    return OpenAI(api_key=api_key)

def with_retries(func, *args, max_tries=3, base_delay=1.0, **kwargs):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt >= max_tries:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)) + 0.05 * attempt)
    raise last_err

# -----------------------------
# OpenAI Ops
# -----------------------------
def chat_json(prompt: str, model: str, max_tokens: int = MAX_TOKENS_COMPLETIONS, temperature: float = 0.2) -> Any:
    client = get_openai_client()
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=REQUEST_TIMEOUT_SECS,
        )
    resp = with_retries(_call)
    content = ""
    try:
        content = resp.choices.message.content or ""
    except Exception:
        content = str(resp)
    obj = extract_first_json_block(content)
    if obj is None:
        # Final cleanup attempt
        obj = extract_first_json_block(strip_code_fences(content))
    return obj

def tts_bytes(text: str, voice: str = "alloy") -> Optional[bytes]:
    client = get_openai_client()
    def _call():
        return client.audio.speech.create(model=TTS_MODEL, voice=voice, input=text, timeout=REQUEST_TIMEOUT_SECS)
    try:
        res = with_retries(_call)
        # Different SDK shapes; try content or read()
        if hasattr(res, "content") and isinstance(res.content, (bytes, bytearray)):
            return bytes(res.content)
        if hasattr(res, "read"):
            return res.read()
        if isinstance(res, dict):
            audio_b64 = res.get("audio") or res.get("data") or res.get("content")
            if isinstance(audio_b64, str):
                return base64.b64decode(audio_b64)
        return None
    except Exception:
        return None

def whisper_transcribe_wav_bytes(wav_bytes: bytes) -> Optional[str]:
    client = get_openai_client()
    try:
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"
        def _call():
            return client.audio.transcriptions.create(model=WHISPER_MODEL, file=bio, response_format="text", timeout=REQUEST_TIMEOUT_SECS)
        res = with_retries(_call)
        if isinstance(res, str):
            return res
        if hasattr(res, "text"):
            return res.text
        if isinstance(res, dict) and "text" in res:
            return res["text"]
        return str(res)
    except Exception:
        return None

# -----------------------------
# Resume Extraction
# -----------------------------
def extract_resume_text(file) -> Optional[str]:
    try:
        filename = getattr(file, "name", "upload")
        mime = getattr(file, "type", "").lower()
        if mime == "application/pdf" or str(filename).lower().endswith(".pdf"):
            reader = PdfReader(file)
            text_chunks = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t:
                    text_chunks.append(t)
            text = "\n".join(text_chunks).strip()
            return text or None
        elif mime == "text/plain" or str(filename).lower().endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore").strip()
        else:
            return None
    except Exception:
        return None

# -----------------------------
# Audio Frames -> WAV
# -----------------------------
def audio_frames_to_wav_bytes(frames) -> Optional[bytes]:
    # frames: sequence of av.AudioFrame from streamlit-webrtc
    # Goal: PCM16, sample rate determined from frames (fallback 48000)
    if not frames:
        return None
    arrays = []
    sample_rate = None
    for f in frames:
        try:
            arr = f.to_ndarray()
            # shape can be (channels, samples) or (samples, channels)
            if arr.ndim == 2:
                # Normalize to (samples, channels)
                if arr.shape <= 2 and arr.shape < arr.shape[21]:
                    arr = arr.T
            elif arr.ndim == 1:
                arr = arr[:, None]
            arrays.append(arr)
            sr = getattr(f, "sample_rate", None) or getattr(f, "rate", None)
            if sr:
                sample_rate = sr
        except Exception:
            continue
    if not arrays:
        return None
    data = np.concatenate(arrays, axis=0)
    if np.issubdtype(data.dtype, np.floating):
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    else:
        data = data.astype(np.int16, copy=False)
    nch = data.shape[21] if data.ndim == 2 else 1
    sr = int(sample_rate or 48000)

    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()

# -----------------------------
# Proctoring Video Processor
# -----------------------------
class ProctorProcessor(VideoProcessorBase):
    def __init__(self, interval: int = DEFAULT_SNAPSHOT_INTERVAL):
        super().__init__()
        self.last = time.time()
        self.interval = max(3, int(interval))

    def recv(self, frame):
        try:
            now = time.time()
            if now - self.last >= self.interval:
                st.session_state.proctoring_img = frame.to_image()
                self.last = now
        except Exception:
            pass
        return frame

# -----------------------------
# PDF Report
# -----------------------------
class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AI Interview Report", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_pdf(name: str, role: str, summary: Dict[str, Any], questions: List[Dict[str, Any]], answers: List[Dict[str, Any]], snapshots: List[bytes]) -> bytes:
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Candidate: {name}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Role: {role}", ln=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Timestamp: {ts}", ln=True)
    pdf.cell(0, 10, f"Overall Score: {summary.get('overall_score','N/A')}/10", ln=True)
    pdf.multi_cell(0, 8, f"Recommendation: {summary.get('recommendation','N/A')}")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Strengths:", ln=True)
    pdf.set_font("Arial", "", 12)
    for s in summary.get("strengths", []) or []:
        pdf.multi_cell(0, 7, f"- {s}")
    pdf.ln(2)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Weaknesses:", ln=True)
    pdf.set_font("Arial", "", 12)
    for w in summary.get("weaknesses", []) or []:
        pdf.multi_cell(0, 7, f"- {w}")
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Q&A Details:", ln=True)
    pdf.set_font("Arial", "", 12)
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        pdf.multi_cell(0, 8, f"Q{i}: {q.get('text','')}")
        pdf.multi_cell(0, 8, f"Answer: {a.get('answer','')}")
        pdf.multi_cell(0, 8, f"Feedback: {a.get('feedback','')} (Score: {a.get('score','N/A')}/10)")
        pdf.ln(2)
    # Embed snapshots (if any) as small images
    if snapshots:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Proctoring Snapshots:", ln=True)
        x, y = 10, 25
        w, h = 60, 45
        for idx, b in enumerate(snapshots):
            try:
                # Write to temp in-memory file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(b)
                    tmp.flush()
                    path = tmp.name
                pdf.image(path, x=x, y=y, w=w, h=h)
                x += w + 5
                if x + w > 190:
                    x = 10
                    y += h + 5
            except Exception:
                continue
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# -----------------------------
# Question Generation / Evaluation / Summary
# -----------------------------
def generate_questions(resume: str, role: str, num: int, model: str) -> Optional[List[Dict[str, Any]]]:
    prompt = PROMPT_TEMPLATES["questions"].format(num=num, role=role, resume=resume)
    obj = chat_json(prompt, model=model)
    qs = validate_questions(obj)
    return qs

def evaluate_answer(resume: str, question: str, answer: str, model: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATES["evaluate"].format(resume=resume, question=question, answer=answer)
    obj = chat_json(prompt, model=model)
    valid = validate_eval(obj) or {"score": 0, "feedback": "No evaluation returned
