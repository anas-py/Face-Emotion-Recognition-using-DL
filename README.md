<div align="center">

<h1>😊 Face Emotion Recognition Using Deep Learning</h1>

<p><em>A real-time facial emotion recognition system powered by ConvNeXt transfer learning and MTCNN face detection — deployed as an interactive Streamlit web application.</em></p>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![ConvNeXt](https://img.shields.io/badge/ConvNeXt-Transfer%20Learning-FF6F00?style=flat-square)
![MTCNN](https://img.shields.io/badge/MTCNN-Face%20Detection-6A0DAD?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-79%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Description](#-description)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Comparison](#-model-comparison)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Emotion Classes](#-emotion-classes)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Application Usage](#-application-usage)
- [Academic Information](#-academic-information)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project presents a **production-ready Face Emotion Recognition (FER) system** that detects and classifies human facial emotions in real time using state-of-the-art deep learning techniques. It supports both **static image uploads** and **live webcam input**, making it suitable for a wide range of practical applications including human-computer interaction, mental health monitoring, educational analytics, and customer sentiment analysis.

The system is built on **ConvNeXt**, a modern convolutional architecture fine-tuned on the **FER2013+ dataset**, and uses **MTCNN** for robust multi-face detection and alignment. The entire solution is packaged as a user-friendly **Streamlit web application** that runs on CPU-based systems without requiring specialized hardware.

---

## 📝 Description

Facial expressions are one of the most natural and universal forms of human communication. Automating their recognition has significant implications across industries — from adaptive e-learning platforms that gauge student engagement, to healthcare systems that detect emotional distress, to retail analytics measuring customer satisfaction.

This project tackles the **Facial Emotion Recognition (FER)** problem as a multi-class image classification task across **8 emotion categories**: Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

**What makes this project stand out:**

- **Systematic model benchmarking** — Four architectures (Custom CNN, MobileNetV3, EfficientNet-B2, and ConvNeXt) were trained and rigorously compared to identify the best-performing model.
- **Robust preprocessing** — A dedicated data preprocessing pipeline handles noise, duplicates, class imbalance, and augmentation before any model sees the data.
- **Production-aware design** — The final app works on CPU-only machines, making it deployable without cloud GPU infrastructure.
- **Multi-face support** — MTCNN detects and processes multiple faces in a single frame simultaneously, each labeled with an emotion and confidence score.
- **Confidence visualization** — Per-emotion probability bar charts give interpretable, transparent predictions beyond a single label.
- **Academic rigor** — The project is backed by a full academic report covering methodology, experiments, results, and analysis.

> This system was developed as part of an M.Sc. Artificial Intelligence & Machine Learning dissertation at **Jamia Millia Islamia, New Delhi**, under the supervision of **Prof. Jahiruddin**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 😄 **8-Class Emotion Detection** | Classifies Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| 👥 **Multi-Face Detection** | Detects and labels multiple faces in a single image or frame simultaneously |
| 📷 **Dual Input Modes** | Supports both static image uploads and live webcam streaming |
| 📊 **Confidence Visualization** | Displays per-emotion probability bar charts alongside predictions |
| 🔄 **Real-Time Inference** | Low-latency predictions suitable for live webcam use |
| 🧠 **ConvNeXt Transfer Learning** | Fine-tuned ImageNet-pretrained model for superior feature extraction |
| 🖥️ **CPU Compatible** | Runs on standard machines without GPU requirements |
| 🌐 **Streamlit Web App** | Clean, interactive UI deployable locally or on the cloud |
| 📈 **Model Benchmarking** | Compared 4 architectures; best model selected based on validation accuracy |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│               Face Emotion Recognition Pipeline                     │
│                                                                     │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │    Input    │───▶│   MTCNN Face     │───▶│   Preprocessing    │  │
│  │ Image/Webcam│    │   Detection &    │    │  Resize · Normalize│  │
│  └─────────────┘    │   Alignment      │    │  RGB · Augmentation│  │
│                     └──────────────────┘    └────────┬───────────┘  │
│                                                      │              │
│  ┌───────────────────────────────────┐   ┌───────────▼───────────┐  │
│  │         Output Layer              │◀──│  ConvNeXt Backbone    │  │
│  │  Emotion Label + Confidence Score │   │  (Fine-tuned on       │  │
│  │  Probability Bar Chart            │   │   FER2013+)           │  │
│  └───────────────────────────────────┘   └───────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  Streamlit Web Interface                     │    │
│  │        Image Upload  ·  Webcam Mode  ·  Results Panel       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.9+ | Core development |
| **Deep Learning** | PyTorch | Model training & inference |
| **Model Architecture** | ConvNeXt (Torchvision) | Emotion classification backbone |
| **Face Detection** | MTCNN (facenet-pytorch) | Multi-face detection & alignment |
| **Computer Vision** | OpenCV | Image & video frame processing |
| **Web Application** | Streamlit | Interactive deployment UI |
| **Data Processing** | NumPy, Pandas | Array ops & data handling |
| **Visualization** | Matplotlib, Seaborn | Training plots & confusion matrix |
| **Dataset** | FER2013+ | 8-class facial emotion dataset |

---

## 📂 Dataset

The model is trained on the **FER2013+** dataset — an improved version of the original FER2013 benchmark with relabeled annotations and an additional **Contempt** class.

| Property | Details |
|---|---|
| **Source** | FER2013+ (Microsoft Research) |
| **Total Samples** | ~35,000+ facial images |
| **Image Format** | 48×48 grayscale (converted to RGB) |
| **Classes** | 8 (Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| **Split** | Train / Validation / Test |
| **Challenge** | Class imbalance — Happy & Neutral dominate; Contempt & Disgust underrepresented |

> **Handling Class Imbalance:** The preprocessing pipeline applies data augmentation (flipping, rotation, scaling) to minority classes and uses weighted loss functions during training to penalize misclassification of underrepresented emotions.

---

## 🤖 Model Comparison

Four deep learning architectures were systematically trained and evaluated on the same dataset split:

| Model | Validation Accuracy | Notes |
|---|---|---|
| Custom CNN | ~69% | Lightweight baseline; limited capacity for complex features |
| MobileNetV3 | ~74% | Fast & mobile-friendly; moderate accuracy |
| EfficientNet-B2 | ~77% | Good accuracy-efficiency trade-off |
| **ConvNeXt (Fine-Tuned)** | **~79% ⭐** | Best performance; selected as final model |

### Why ConvNeXt?

ConvNeXt is a **modern pure-CNN architecture** inspired by the design principles of Vision Transformers but retaining the computational efficiency of convolutions. Key advantages:

- Stronger hierarchical feature extraction than older CNNs
- Better generalization from large-scale ImageNet pretraining
- No self-attention overhead — faster inference on CPU
- Fine-tuning the later layers on FER2013+ preserves low-level visual features while adapting high-level emotion-specific representations

---

## ⚙️ Preprocessing Pipeline

A dedicated `data-preprocessing.ipynb` notebook handles all data cleaning and preparation steps before model training:

```
Raw FER2013+ Images
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: Duplicate Removal       → Perceptual hashing (near-identical │
│                                    images removed)                    │
│  Step 2: Image Resizing          → Standardized to model input dims   │
│  Step 3: Grayscale → RGB         → 3-channel conversion for pretrained│
│                                    model compatibility                │
│  Step 4: Histogram Equalization  → Contrast improvement under varied  │
│                                    lighting conditions                │
│  Step 5: Normalization           → ImageNet mean & std                │
│                                    μ=[0.485, 0.456, 0.406]            │
│                                    σ=[0.229, 0.224, 0.225]            │
│  Step 6: Data Augmentation       → Random flips, rotations, scaling   │
│                                    applied to minority emotion classes │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
   Clean, Balanced Dataset → Model Training
```

---

## 😃 Emotion Classes

The system recognizes **8 universal facial emotion categories**:

| # | Emotion | Visual Cues |
|---|---|---|
| 1 | 😠 Angry | Furrowed brows, tense facial muscles, narrowed eyes |
| 2 | 😒 Contempt | Asymmetric lip raise, one-sided expression |
| 3 | 🤢 Disgust | Nose wrinkle, upper lip raise, chin raised |
| 4 | 😨 Fear | Wide eyes, raised brows, open mouth, tense |
| 5 | 😄 Happy | Raised cheeks, lip corners pulled back and up |
| 6 | 😐 Neutral | Relaxed muscles, no dominant expression |
| 7 | 😢 Sad | Drooping lip corners, inner brow raise, glassy eyes |
| 8 | 😲 Surprise | Raised brows, very wide eyes, open mouth |

---

## 📈 Results & Performance

| Metric | Value |
|---|---|
| **Best Model** | ConvNeXt (Fine-Tuned) |
| **Validation Accuracy** | ~79% |
| **Dataset** | FER2013+ |
| **Input Resolution** | 48×48 → upscaled for ConvNeXt |
| **Inference Mode** | CPU-compatible real-time |
| **Emotion Classes** | 8 |

**Confusion Matrix Highlights:**
- ✅ **Strong performance** on Happy and Neutral — the most represented classes in the dataset
- ✅ **Reliable detection** of Surprise and Angry due to distinctive visual features
- ⚠️ **Lower recall** on Contempt and Disgust due to dataset imbalance and high inter-class visual similarity
- 🔄 Augmentation and weighted loss partially mitigate imbalance; future work could explore SMOTE-style oversampling

> 📄 Full training curves, confusion matrices, and per-class metrics are documented in [`REPORT_git.pdf`](./REPORT_git.pdf).

---

## 📁 Project Structure

```
Face-Emotion-Recognition-using-DL/
│
├── app.py                               # Streamlit web application (main entry point)
├── data-preprocessing.ipynb            # Data cleaning, augmentation & preparation
├── face-emotions-recognition-fer.ipynb  # ConvNeXt model training & evaluation
├── efficientnet_b2.ipynb               # EfficientNet-B2 experiment notebook
├── requirements.txt                    # Python dependencies
├── REPORT_git.pdf                      # Full academic project report
├── output.png                          # Sample prediction output image
│
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- Webcam (optional, for live detection mode)
- Virtual environment (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/anas-py/Face-Emotion-Recognition-using-DL.git
cd Face-Emotion-Recognition-using-DL
```

### 2. Create & Activate a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 🖥️ Application Usage

Once the app is running, you can interact with it in two modes:

### 📸 Image Upload Mode
1. Click **"Upload Image"** and select a `.jpg` or `.png` file
2. MTCNN automatically detects all faces in the uploaded image
3. Each detected face is annotated with an emotion label and confidence score (%)
4. A probability bar chart displays the distribution across all 8 emotion classes

### 🎥 Webcam Mode
1. Click **"Start Webcam"** to enable live detection
2. The system processes each video frame in real time
3. Detected faces are highlighted with bounding boxes and emotion labels
4. Confidence scores update live per frame
5. Works seamlessly on standard laptop webcams — no GPU required

---

## 🎓 Academic Information

This project was developed as part of a formal academic dissertation:

| Field | Details |
|---|---|
| **Author** | Mohd Anas |
| **Degree** | M.Sc. Artificial Intelligence & Machine Learning |
| **Semester** | III (2025–26) |
| **Supervisor** | Prof. Jahiruddin |
| **University** | Jamia Millia Islamia, New Delhi |

📄 The complete academic report — including literature review, methodology, model experiments, results, and future work — is available in [`REPORT_git.pdf`](./REPORT_git.pdf).

---

## 🤝 Contributing

Contributions are welcome! Suggested improvements include:

- Adding Vision Transformer (ViT) or Swin Transformer as additional backbone options
- Implementing SMOTE-based oversampling for minority emotion classes
- Adding video file input support alongside webcam
- Deploying to **Streamlit Cloud** or **HuggingFace Spaces** for public access
- Adding emotion trend tracking over time in webcam mode

**To contribute:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add: descriptive commit message'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow PEP 8 style guidelines and include docstrings for any new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [anas-py](https://github.com/anas-py) · Jamia Millia Islamia, New Delhi

⭐ If you found this project helpful, consider giving it a star!

</div>
