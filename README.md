# File 3: README.md

# AI Interviewer Streamlit App

This app provides a full AI-driven interview experience with login/authentication, resume upload, AI question generation, voice question & answer, GPT-based evaluation, and comprehensive reporting.

## Features

- Login & authentication with `streamlit-authenticator`
- Upload resume (PDF/TXT) or use demo resume
- Generate personalized interview questions (GPT-4o/4/3.5)
- Answer questions using microphone with speech-to-text (Whisper API)
- Text-to-speech question playback
- GPT scoring, feedback, improvements & model answers
- Download PDF report & JSON session export
- Save sessions with SQLite and local JSON files

## Setup


## Configuration

- Set your OpenAI API key either in environment variable `OPENAI_API_KEY` or paste in sidebar input.

## Usage

- Login with demo credentials:
  - alice / 123
  - bob / abc
  - carol / xyz
- Upload your resume or select demo
- Enter candidate info, start interview
- Answer questions using voice or fallback text
- Download detailed PDF and JSON report
- Save session for future review

## License

Open source for demonstration purposes.

---

Developed by
