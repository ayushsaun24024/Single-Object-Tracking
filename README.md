UAV Single Object Tracker

Deep learning-based single object tracking system for UAV videos using hybrid ML models with camera motion compensation.

Live Demo: [Coming Soon - HuggingFace Spaces]

Features

- Single object tracking in UAV/drone videos
- Camera motion compensation using ORB features
- Multi-scale sliding window search with SIFT
- Hybrid ML models (Linear Regression + Random Forest)
- Gradio web interface for easy inference
- Pre-trained models included

Project Structure

```
uav-object-tracker/
├── train.py              - Training pipeline
├── inference.py          - Video inference script
├── app.py               - Gradio web interface
├── requirements.txt     - Python dependencies
├── models/              - Pre-trained model weights
│   ├── position_model.joblib
│   ├── size_model.joblib
│   ├── position_scaler.joblib
│   └── size_scaler.joblib
└── README.md
```

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/uav-object-tracker.git
cd uav-object-tracker

Install dependencies:

pip install -r requirements.txt

Usage

Local Inference

from inference import ObjectTrackerInference

tracker = ObjectTrackerInference(model_dir='models')
tracker.track_video('input.mp4', initial_bbox=[100, 100, 50, 50], output_path='output.mp4')

Web Interface (Gradio)

python app.py

Then open http://localhost:7860 in your browser.

Training

To train your own models on custom dataset:

python train.py

Make sure your dataset follows the structure:

```
dataset/
├── sequences/       - Video frames
└── annotations/     - Bounding box annotations (.txt files)
```

Dataset Format

Each annotation file contains per-frame bounding boxes:

x, y, width, height

Where:
- x, y: Top-left corner coordinates
- width, height: Bounding box dimensions

Model Architecture

Position Prediction: Linear Regression
Size Prediction: Random Forest Regressor (150 trees)

Features:
- HOG descriptors (64 features)
- Local Binary Patterns (5 features)
- Camera motion (4 features)
- Position/size info (4 features)

Deployment

HuggingFace Spaces

This project is deployed on HuggingFace Spaces for free public access.

Visit: [YOUR_SPACE_URL]

Requirements

- Python 3.8+
- OpenCV 4.8+
- scikit-learn 1.3+
- Gradio 4.19+

Results

Mean IoU: 0.XX
Position MAE: XX pixels
Size MAE: XX pixels

License

MIT License - feel free to use for research and commercial projects.

Citation

If you use this project, please cite:

@misc{uav-object-tracker,
  author = {YOUR_NAME},
  title = {UAV Single Object Tracker},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/uav-object-tracker}
}

Acknowledgments

- Dataset based on UAV tracking sequences
- Built with OpenCV, scikit-learn, and Gradio
- Deployed on HuggingFace Spaces

📁 Structure After This Step:

```
uav-object-tracker/          (root)
├── train.py                 
├── inference.py             
├── app.py                   
├── requirements.txt         
├── README.md                ← NEW (place here)
└── models/                  
    ├── position_model.joblib
    ├── size_model.joblib
    ├── position_scaler.joblib
    └── size_scaler.joblib

```