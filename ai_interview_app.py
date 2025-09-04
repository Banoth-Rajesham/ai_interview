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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import time

# --- Page Config ---
st.set_page_config(page_title="🧠 AI Interviewer", layout="wide", page_icon="🧠")

# --- Constants ---
MODELS = {"GPT-4o": "gpt-4o", "GPT-4": "gpt-4", "GPT-3.5": "gpt-3.5-turbo"}
SESSION_DIR = "saved_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Top-Level Class Definition for WebRTC ---
# The InterviewProcessor class is defined here to ensure it's stable across reruns.
class InterviewProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.last_proctor_time = time.time()

    def recv(self, frame):
        if isinstance(frame, av.AudioFrame):
            self.audio_buffer.append(frame.to_ndarray().tobytes())
            return frame
        elif isinstance(frame, av.VideoFrame):
            if time.time() - self.last_proctor_time > 10:
                st.session_state.proctoring_img = frame.to_image()
                self.last_proctor_time = time.time()
            return frame```


# --- User Authentication ---
if not os.path.exists('config.yaml'):
    st.error("Fatal Error: `config.yaml` not found. Please create the configuration file.")
    st.stop()

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
