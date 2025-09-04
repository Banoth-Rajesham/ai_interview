import streamlit as st
import openai
import PyPDF2
import io
import base64

# This file contains helper functions.

def get_openai_key():
    key = st.session_state.get("openai_api_key", "")
    if not key and "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    if not key:
        st.error("Please add your OpenAI API key in the sidebar!")
        st.stop()
    return key

def openai_client():
    key = get_openai_key()
    return openai.OpenAI(api_key=key)

def chat_completion(messages, model="gpt-4o", temperature=0.3, max_tokens=1500):
    client = openai_client()
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
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
        st.warning(f"TTS Error: {e}")
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
        text = "".join(page.extract_text() or "" for page in PyPDF2.PdfReader(file).pages)
    elif file.name.lower().endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        return None
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    md = f"""<audio controls autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></source></audio>"""
    st.markdown(md, unsafe_allow_html=True)
