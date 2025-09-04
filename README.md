# AI Interviewer Streamlit App

This app provides a full AI-driven interview experience with user login, resume upload, AI-generated questions, voice answers, and comprehensive reporting.

## Features

- **User Authentication:** Secure login and registration powered by `streamlit-authenticator`.
- **Resume Parsing:** Upload a resume in PDF or TXT format to tailor the interview.
- **Dynamic Question Generation:** Uses OpenAI's GPT models to create relevant questions based on the resume and job role.
- **Voice-to-Text Answers:** Candidates can answer questions using their voice, which is transcribed in real-time.
- **Video & Audio Streaming:** Utilizes `streamlit-webrtc` for live video feed and audio capture.
- **AI-Powered Evaluation:** Candidate answers are evaluated by an AI, providing a score and constructive feedback.
- **Proctoring:** Captures periodic snapshots from the video feed to ensure the candidate's presence.
- **PDF Reports:** Generates a downloadable PDF summary of the entire interview session.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai_interview.git
    cd ai_interview
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Credentials:**
    - Create a `config.yaml` file with the structure provided.
    - **Important:** You must hash your passwords before adding them. Create and run a temporary Python script (`generate_keys.py`) with `import streamlit_authenticator as stauth; print(stauth.Hasher(['pass1', 'pass2']).generate())` to get the hashed values.

4.  **Run the application:**
    ```bash
    streamlit run ai_interview_app.py
    ```

## File Structure

-   `ai_interview_app.py`: The main Streamlit application script.
-   `config.yaml`: Stores user credentials and authentication settings.
-   `requirements.txt`: Lists all required Python packages.
-   `README.md`: This file.
