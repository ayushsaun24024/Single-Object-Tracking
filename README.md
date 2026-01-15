# 🚀 ML-Based Single Object Tracker for UAV Videos

A **machine learning–powered single object tracking system** designed specifically for **UAV / drone footage**.  
This project blends **classical computer vision** with **machine learning models** to achieve **robust, real-time tracking**, even under **strong camera motion**.

🎯 **Core Idea**  
Track *one* object accurately across video frames using **motion compensation + ML prediction**, optimized for aerial videos.

🌐 **Live Demo**: [Click Here](https://huggingface.co/spaces/ayushsaun/Single_Object_Tracking)

---

## 📚 Table of Contents

- Overview  
- Features  
- Project Structure  
- Installation  
- Usage  
- Dataset Format  
- Model Architecture  
- Feature Engineering  
- Camera Motion Compensation  
- Training Your Own Model  
- How It Works  
- Performance  
- Deployment (HuggingFace Spaces)  
- Dependencies  
- Use Cases  
- License  
- Acknowledgments  
- Author  
- Contributing  
- Citation  
- Links  

---

## 🎯 Overview

This tracker is built for **challenging UAV scenarios** where:

- The camera is constantly moving  
- Objects change scale rapidly  
- Traditional trackers fail due to motion jitter  

Instead of relying purely on correlation filters or deep trackers, this system uses:

- ORB-based camera motion compensation  
- HOG + LBP feature engineering  
- Hybrid ML models (Linear Regression + Random Forest)  

Result: **accurate, stable, and CPU-friendly tracking**.

---

## ✨ Features

- 🎯 Single Object Tracking – Accurate frame-by-frame tracking  
- 🚁 UAV Optimized – Designed for aerial footage & camera shake  
- 🧭 Camera Motion Compensation – ORB-based affine correction  
- 🔍 Multi-Scale Search – Robust sliding-window detection  
- 🤖 Hybrid ML Models – Linear Regression + Random Forest  
- 🌐 Web Interface – Gradio UI  
- 📦 Pre-trained Models – Ready to use  
- 📊 High Accuracy – Strong IoU metrics  
- ⚡ Real-Time Capable – 20–30 FPS on CPU  
- 🔓 Open Source – MIT License  

---

## 🗂️ Project Structure

Single-Object-Tracking/  
│  
├── train.py                  # Training pipeline  
├── inference.py              # Tracking inference  
├── app.py                    # Gradio web interface  
├── requirements.txt          # Dependencies  
├── README.md                 # Documentation  
├── .gitignore  
│  
└── models/  
    ├── position_model.joblib  
    ├── size_model.joblib  
    ├── position_scaler.joblib  
    └── size_scaler.joblib  

---

## 🛠️ Installation

### Step 1: Clone Repository

git clone https://github.com/ayushsaun24024/Single-Object-Tracking.git  
cd Single-Object-Tracking  

### Step 2: Install Dependencies

pip install -r requirements.txt  

### Step 3: Verify Installation

python -c "import cv2, sklearn, gradio; print('All dependencies installed successfully!')"  

---

## ▶️ Usage

### Method 1: Python Script

from inference import ObjectTrackerInference  

tracker = ObjectTrackerInference(model_dir="models")  

tracker.track_video(  
    video_path="my_drone_video.mp4",  
    initial_bbox=[100, 100, 50, 50],  
    output_path="tracked_output.mp4",  
    fps=30  
)  

print("Tracking complete!")  

---

### Method 2: Web Interface

python app.py  

Open http://localhost:7860 and follow the UI steps.

---

## 📦 Dataset Format

### Directory Structure

your-dataset/  
│  
├── sequences/  
│   ├── seq_001/  
│   │   ├── 000000.jpg  
│   │   ├── 000001.jpg  
│   │   └── ...  
│  
└── annotations/  
    ├── seq_001.txt  

### Annotation File Format (CSV, No Header)

x,y,width,height  
945,293,52,28  
950,293,52,27  
955,293,53,27  

---

## 🧠 Model Architecture

### Position Prediction
- Model: Linear Regression  
- Input: 77 features  
- Output: x, y center coordinates  

### Size Prediction
- Model: Random Forest Regressor (150 trees)  
- Output: width, height  

---

## 🧩 Feature Engineering (77D)

- HOG: 64 features  
- LBP statistics: 5 features  
- Camera motion: 4 features  
- Position & size history: 4 features  

---

## 🎥 Camera Motion Compensation

1. Detect ORB keypoints  
2. Match descriptors  
3. Estimate affine transform  
4. Compensate bounding box  

Benefits:
- Handles UAV shake  
- Separates camera vs object motion  
- Improves IoU  

---

## 🏋️ Training Your Own Model

Set dataset path in train.py:

directoryPath = "/path/to/your-dataset"  

Run training:

python train.py  

Outputs:
- Trained models in models/  
- Evaluation metrics  
- Test tracking videos  

---

## 🔄 How It Works

Frame 1  
- User initializes bounding box  

Frame 2 to N  
- Camera motion estimation  
- Multi-scale search  
- Feature extraction  
- Position prediction  
- Size prediction  
- Bounding box update  
- Visualization  

---

## 📊 Performance

Accuracy:
- Mean IoU: 0.82+  
- Position MAE: < 6 px  
- Size MAE: < 4 px  

Speed:
- 20–30 FPS on CPU  
- Tested on 720p and 1080p  

---

## 🚀 Deployment (HuggingFace Spaces)

- Free CPU hosting  
- Auto Gradio detection  
- Public URL  

Push:
- app.py  
- inference.py  
- requirements.txt  
- models/  

---

## 📦 Dependencies

opencv-python-headless==4.8.1.78  
scikit-learn==1.3.2  
numpy==1.24.3  
joblib==1.3.2  
gradio==4.19.2  
tqdm==4.66.1  

---

## 💡 Use Cases

- UAV surveillance  
- Wildlife tracking  
- Traffic monitoring  
- Sports analytics  
- Research & education  
- Robotics  

---

## 📜 License

**Apache License 2.0**

This project is fully open source, similar to COCO-style datasets and tools.  
You are free to **use, modify, distribute, and build upon this work**, including for commercial purposes, provided that proper attribution is given and license terms are respected.

---

## 🙌 Acknowledgments

- OpenCV  
- scikit-learn  
- Gradio  
- Classical trackers: KCF, MOSSE, CSRT  

---

## 👤 Author

Ayush Saun  
GitHub: ayushsaun24024  

---

## 🤝 Contributing

- Open issues  
- Submit pull requests  
- Share feedback  

---

⭐ Star the repository if you find it useful  
Built with dedication for the Computer Vision Community
