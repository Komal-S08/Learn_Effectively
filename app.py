# 📦 Smart Course Completion Assistant with Voice, Gemini Chat, Export & Email
//changed file

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import spacy
import base64
from datetime import datetime
from streamlit_chat import message
from fpdf import FPDF
import speech_recognition as sr
import google.generativeai as genai
import resend

# --------------------🔐 Configuration --------------------

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Configure Gemini
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "AIzaSyCuC6mpVuXTCn2I8JsWTkmCzhYSMwkfRYY"))

# Configure Resend
resend.api_key = st.secrets.get("RESEND_API_KEY", "re_NEaN9Hcg_P6E4fvBTnA1UAUevg6ZmjwPG")

# --------------------⚙️ Streamlit App --------------------

st.set_page_config(page_title="Smart Course Assistant", layout="centered")
st.title("🎓 Smart Course Completion Assistant")
st.markdown("Ask in natural language or speak. I’ll predict and explain course completion.")

# --------------------💬 Session Setup --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "predictions" not in st.session_state:
    st.session_state.predictions = []

# --------------------🎙️ Voice Input --------------------

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... speak now")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError:
        return "Speech service unavailable."

# --------------------🔍 Feature Extraction --------------------

def extract_features(text):
    doc = nlp(text.lower())
    data = {
        "course_category": 0, "device_type": 1, "time_spent": 0.0,
        "videos_watched": 0, "quizzes_taken": 0, "quiz_scores": 0.0,
        "completion_rate": 0.0, "forum_participation": 0, "peer_interaction": 0,
        "feedback_given": 0, "reminders_clicked": 0, "support_usage": 0,
    }
    patterns = {
        "time_spent": r"spent (\d+\.?\d*) hours",
        "videos_watched": r"watched (\d+)",
        "quizzes_taken": r"took (\d+)",
        "quiz_scores": r"scored (\d+\.?\d*)\%",
        "completion_rate": r"completed (\d+\.?\d*)\%",
        "forum_participation": r"forum.*?(\d+)",
        "peer_interaction": r"peer.*?(\d+)",
        "feedback_given": r"feedback.*?(\d+)",
        "reminders_clicked": r"reminder.*?(\d+)",
        "support_usage": r"support.*?(\d+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            data[key] = value if key in ["time_spent", "quiz_scores", "completion_rate"] else int(value)
    return np.array([[data[k] for k in data]]), data

# --------------------💡 Gemini Response --------------------

def ask_gemini(features, prediction, prob):
    prompt = f"""A student has these learning engagement metrics:
{features}

Prediction: {'Completed' if prediction == 1 else 'Not Completed'}
Confidence Score: {prob:.2f}

Explain the prediction in simple words."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# --------------------📤 Export & Email --------------------

def get_chat_df():
    users = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    bots = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]
    return pd.DataFrame({"User": users, "Assistant": bots[:len(users)]})

def export_predictions_to_pdf(predictions, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Course Completion Prediction Report", ln=True, align='C')
    pdf.ln(10)
    for i, pred in enumerate(predictions):
        pdf.multi_cell(0, 10, txt=f"Prediction #{i+1}: {'Completed' if pred == 1 else 'Not Completed'}")
        pdf.ln(5)
    pdf.output(filename)

def send_email_with_attachment(receiver, subject, body, attachment_path):
    with open(attachment_path, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode()
    resend.Emails.send({
        "from": "Assistant <onboarding@resend.dev>",
        "to": receiver,
        "subject": subject,
        "html": f"<p>{body}</p>",
        "attachments": [{
            "filename": attachment_path,
            "content": encoded_file,
            "contentType": "application/octet-stream"
        }]
    })

# --------------------📨 Chat Interface --------------------

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("You:", key="input")
with col2:
    if st.button("🎤 Speak"):
        user_input = record_audio()
        st.session_state.input = user_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    features, raw_data = extract_features(user_input)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    st.session_state.predictions.append(pred)
    response = ask_gemini(raw_data, pred, prob)
    st.session_state.messages.append({"role": "assistant", "content": response})
    message(response, is_user=False, key=str(len(st.session_state.messages)))

# --------------------📊 Sidebar Tools --------------------

st.sidebar.markdown("## 📤 Export Tools")

# CSV
if st.sidebar.button("💾 Save Chat as CSV"):
    df = get_chat_df()
    fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False)
    st.sidebar.success(f"Saved: {fname}")

# PDF
if st.sidebar.button("🧾 Generate PDF Report"):
    export_predictions_to_pdf(st.session_state.predictions)
    with open("report.pdf", "rb") as f:
        st.sidebar.download_button("⬇️ Download PDF", f, "report.pdf")

# Email
email = st.sidebar.text_input("📧 Email Address")
if st.sidebar.button("📨 Send Report via Email"):
    df = get_chat_df()
    fname = "chat_report.csv"
    df.to_csv(fname, index=False)
    send_email_with_attachment(email, "Assistant Chat Report", "Here is your session log.", fname)
    st.sidebar.success("📤 Email sent!")

# Stats
st.sidebar.markdown("## 📊 Completion Stats")
total = len(st.session_state.predictions)
completed = st.session_state.predictions.count(1)
not_completed = st.session_state.predictions.count(0)
st.sidebar.write(f"✅ Completed: {completed}")
st.sidebar.write(f"❌ Not Completed: {not_completed}")
if total > 0:
    fig, ax = plt.subplots()
    ax.pie([completed, not_completed], labels=["Completed", "Not Completed"],
           colors=["#66b3ff", "#ff9999"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.sidebar.pyplot(fig)
