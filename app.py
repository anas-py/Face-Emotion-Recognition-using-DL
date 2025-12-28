import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
import matplotlib.pyplot as plt

# ===== Project Utilities =====
from utils.model_loader import load_model
from utils.face_utils import detect_faces
from utils.preprocess import get_transform, predict_emotion


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Face Emotion Recognition Using Deep Learning",
    page_icon="😊",
    layout="centered"
)


# ==================================================
# THEME HANDLER (DARK / LIGHT)
# ==================================================
def set_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: white; }
            h1, h2, h3, h4, h5, h6, p, label { color: white; }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: white; color: black; }
            </style>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# LOAD MODEL
# ==================================================
MODEL_PATH = "best_emotion_model_convnext_v2.pth"
model, CLASS_NAMES, IMG_SIZE, DEVICE = load_model(MODEL_PATH)
transform = get_transform(IMG_SIZE)


# ==================================================
# SESSION STATE (EMOTION STATS)
# ==================================================
if "emotion_stats" not in st.session_state:
    st.session_state.emotion_stats = {e: 0 for e in CLASS_NAMES}


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Image Emotion Detection",
        "Video Emotion Detection",
        "Webcam Emotion Detection",
        "Emotion Statistics",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("🎨 **Theme Settings**")
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1)
set_theme(theme)


# ==================================================
# PROBABILITY BAR CHART
# ==================================================
def show_probability_chart(probs):
    fig, ax = plt.subplots()
    ax.barh(CLASS_NAMES, probs)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Probability Distribution")
    st.pyplot(fig)


# ==================================================
# FRAME PROCESSING
# ==================================================
def process_frame(frame, show_probs=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)
        face_tensor = transform(face_pil)

        with torch.no_grad():
            pred, conf, probs = predict_emotion(
                model, face_tensor, DEVICE, return_probs=True
            )

        emotion = CLASS_NAMES[pred]
        label = f"{emotion} ({conf:.1f}%)"

        # Update stats
        st.session_state.emotion_stats[emotion] += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        if show_probs:
            show_probability_chart(probs)

    return frame


# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":
    st.title("😊 Face Emotion Recognition System")
    st.subheader("Using Deep Learning")
    st.write("""
    **Dataset:** FER2013+ (8 Emotions)  
    **Model:** ConvNeXt (Transfer Learning)  
    **Framework:** PyTorch  
    **Face Detection:** MTCNN  
    """)


# ==================================================
# IMAGE MODE
# ==================================================
elif page == "Image Emotion Detection":
    st.header("📷 Image Emotion Recognition")

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    show_probs = st.checkbox("Show Emotion Probability Chart")

    if file:
        image = Image.open(file).convert("RGB")
        frame = np.array(image)
        output = process_frame(frame, show_probs)
        st.image(output, channels="RGB", use_column_width=True)


# ==================================================
# VIDEO MODE
# ==================================================
elif page == "Video Emotion Detection":
    st.header("🎥 Video Emotion Recognition")

    file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    show_probs = st.checkbox("Show Emotion Probability Chart")

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output = process_frame(frame, show_probs)
            stframe.image(output, channels="BGR")

        cap.release()


# ==================================================
# WEBCAM MODE
# ==================================================
elif page == "Webcam Emotion Detection":
    st.header("📡 Real-Time Webcam Emotion Recognition")

    show_probs = st.checkbox("Show Emotion Probability Chart")
    run = st.checkbox("Start Webcam")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        output = process_frame(frame, show_probs)
        stframe.image(output, channels="BGR")

    cap.release()


# ==================================================
# EMOTION STATISTICS
# ==================================================
elif page == "Emotion Statistics":
    st.header("📊 Session Emotion Statistics")

    emotions = list(st.session_state.emotion_stats.keys())
    counts = list(st.session_state.emotion_stats.values())

    fig, ax = plt.subplots()
    ax.bar(emotions, counts)
    ax.set_ylabel("Count")
    ax.set_title("Detected Emotions in Current Session")
    st.pyplot(fig)


# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "About Project":
    st.header("📘 About the Project")

    st.write("""
    **Project Title:** Face Emotion Recognition Using Deep Learning  

    **Student:**  
    Mohd Anas  
    M.Sc. Artificial Intelligence & Machine Learning  

    **Supervisor:**  
    Prof. Jahiruddin  
    Department of Computer Science  
    Jamia Millia Islamia, New Delhi  

    **Description:**  
    This project focuses on recognizing human emotions from facial expressions
    using deep learning techniques. The system supports real-time multi-face
    emotion recognition using a ConvNeXt-based model trained on the FER2013+
    dataset.
    """)


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>© 2025 | Face Emotion Recognition Using Deep Learning</p>",
    unsafe_allow_html=True
)
