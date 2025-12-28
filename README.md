# 😊 Face Emotion Recognition Using Deep Learning

<p align="center">
  <img src="screenshots/streamlit_ui.png" width="80%">
</p>

<p align="center">
  <b>Real-Time Facial Emotion Recognition using ConvNeXt & MTCNN</b><br>
  Deep Learning | Computer Vision | Streamlit Deployment
</p>

---

## 📌 Project Overview

This project presents a **deep learning–based Face Emotion Recognition (FER) system** capable of identifying **human emotions from facial expressions** using both **static images and real-time webcam input**.

The system leverages:
- **ConvNeXt (Transfer Learning)** for high-accuracy emotion classification  
- **MTCNN** for robust multi-face detection  
- **Streamlit** for real-time, interactive deployment  

The model is trained on the **FER2013+ dataset**, which contains **8 facial emotion classes**.

---

## 🎓 Academic Information

- **Student:** Mohd Anas  
- **Roll No:** 24MAM023  
- **Course:** M.Sc. Artificial Intelligence & Machine Learning  
- **Semester:** III (2025–26)  
- **Supervisor:** Prof. Jahiruddin  
- **University:** Jamia Millia Islamia, New Delhi  

---

## 😃 Emotion Classes

The system classifies facial expressions into the following **8 emotions**:

| Emotion |
|-------|
| Angry |
| Contempt |
| Disgust |
| Fear |
| Happy |
| Neutral |
| Sad |
| Surprise |

---

## 🧠 Model Architecture

### 🔹 Baseline Models
- Custom CNN  
- MobileNetV3  
- EfficientNet-B2  

### 🔹 Final Selected Model
✅ **ConvNeXt (Fine-Tuned)**  
- Pretrained on **ImageNet**
- Fine-tuned on **FER2013+**
- Strong hierarchical feature extraction
- Best accuracy vs speed trade-off

---

## 📊 Model Performance

| Model | Validation Accuracy |
|-----|--------------------|
| Custom CNN | ~69% |
| MobileNetV3 | ~74% |
| EfficientNet-B2 | ~77% |
| **ConvNeXt (Fine-Tuned)** | **~79%** ⭐ |

✔ Highest overall performance  
✔ Stable real-time inference  

---

## 📷 Visual Results

### 🔹 Streamlit User Interface
<p align="center">
  <img src="screenshots/streamlit_ui.png" width="75%">
</p>

---

### 🔹 Real-Time Webcam Emotion Detection
<p align="center">
  <img src="screenshots/webcam_detection.png" width="75%">
</p>

✔ Multi-face detection  
✔ Emotion label with confidence (%)  

---

### 🔹 Confusion Matrix
<p align="center">
  <img src="screenshots/confusion_matrix.png" width="60%">
</p>

> Strong performance for **Happy** and **Neutral** emotions.  
> Lower accuracy for **Contempt** and **Disgust** due to dataset imbalance.

---

## 🔍 Face Detection

- **MTCNN (Multi-task Cascaded CNN)**
- Detects **multiple faces simultaneously**
- Performs face alignment
- Robust to lighting and pose variations

---

## ⚙️ Preprocessing Pipeline

✔ Duplicate image removal (hashing)  
✔ Image resizing  
✔ Grayscale → RGB conversion  
✔ Histogram equalization  
✔ Normalization (ImageNet mean & std)  
✔ Data augmentation (flip, rotation, scaling)

---

## ✨ Key Features

✔ Multi-face emotion detection  
✔ Real-time webcam recognition  
✔ Emotion confidence percentage  
✔ Emotion probability visualization  
✔ Image & webcam input support  
✔ User-friendly Streamlit UI  
✔ Works on **CPU-based systems**

---

## 🚀 Deployment (Run Locally)

### 🔹 Step 1: Install Dependencies
```bash
pip install -r requirements.txt
