import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

# --- Import all functions and classes from your other python files ---
from utils import *
from core_ai_logic import *
from pdf_generator import *
from ui_components import *

# --- Top-Level Class Definition for WebRTC ---
# CRITICAL FIX: This class MUST be defined in the main script to be stable.
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
            return frame

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

# --- Main App Execution Logic ---
def app_logic():
    st.title("🧠 AI Interviewer")
    if "stage" not in st.session_state:
        st.session_state.stage = "setup"
    if st.session_state.stage == "setup":
        setup_section(authenticator, config) # Pass authenticator and config
    elif st.session_state.stage == "interview":
        interview_section(authenticator, config, InterviewProcessor) # Pass the class
    elif st.session_state.stage == "summary":
        summary_section(authenticator, config)

# Main entry point for the application
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

if not st.session_state["authentication_status"]:
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        authenticator.login()
        if st.session_state["authentication_status"]:
            st.rerun()
        elif st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password.')

    with register_tab:
        st.subheader("Create a New Account")
        try:
            if authenticator.register_user(fields={'Form name': 'Create Account', 'Username': 'username', 'Name': 'name', 'Email': 'email', 'Password': 'password'}):
                st.success('User registered successfully! Please go to the Login tab to sign in.')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
else:
    sidebar(authenticator) # Pass authenticator
    app_logic()
