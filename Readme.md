# Enhanced Single Object Tracking Implementation

## Overview
This repository implements a robust single object tracking system that combines traditional computer vision techniques with machine learning approaches. The system features camera motion compensation, multi-scale sliding windows, and hybrid prediction models to achieve reliable object tracking across video sequences.

## Key Components

### 1. Camera Motion Compensator
The `CameraMotionCompensator` class handles global camera motion estimation using feature matching:
- Uses ORB (Oriented FAST and Rotated BRIEF) features for efficient detection
- Implements affine transformation estimation for motion modeling
- Employs robust matching with distance-based filtering
- Maintains frame-to-frame correspondence for continuous tracking

### 2. Improved Sliding Window Tracker
The `ImprovedSlidingWindowTracker` class implements an advanced sliding window approach:
- Multi-scale window generation with scale pyramid
- SIFT (Scale-Invariant Feature Transform) feature extraction
- FLANN-based feature matching for efficient similarity computation
- Adaptive window overlap and scale factor parameters
- Score-based window selection mechanism

### 3. Hybrid Tracking Pipeline
The `ImprovedHybridTrackingPipeline` class combines multiple tracking strategies:
- Feature extraction using HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns)
- Separate position and size prediction models
- Camera motion compensation integration
- Template updating mechanism with IoU-based verification
- Comprehensive evaluation metrics

## Technical Details

### Feature Extraction
The system uses multiple feature types:
1. HOG Features:
   - Captures object appearance and shape
   - Uses 64x64 cell structure with 16x16 blocks
   - 9-bin orientation histograms

2. LBP Features:
   - Captures texture information
   - Uses radius=1 and 8 sampling points
   - Implements bilinear interpolation for accurate sampling

3. Motion Features:
   - Incorporates camera motion parameters
   - Includes scale and translation components
   - Enables motion-aware tracking

### Machine Learning Models
The system employs two specialized models:
1. Linear Regression:
   - Used for position prediction (x, y coordinates)
   - Provides fast and stable predictions
   - Handles linear motion patterns

2. Random Forest Regressor:
   - Used for size prediction (width, height)
   - 150 estimators for robust ensemble predictions
   - Handles non-linear size variations

### Performance Metrics
The system evaluates tracking performance using:
- Mean Absolute Error (MAE) for position and size
- Root Mean Square Error (RMSE)
- R² Score for prediction quality
- Intersection over Union (IoU) for overall tracking accuracy
- Per-coordinate performance metrics

## Usage

### Prerequisites
```python
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Directory Structure
```
root_directory/
├── sequences/
│   ├── sequence1/
│   │   ├── frame001.jpg
│   │   ├── frame002.jpg
│   │   └── ...
│   └── sequence2/
└── annotations/
    ├── sequence1.txt
    └── sequence2.txt
```

### Running the Pipeline
```python
pipeline = ImprovedHybridTrackingPipeline(
    directoryPath='path/to/data',
    test_size=0.2,
    random_state=42
)
results = pipeline.train_and_evaluate(output_dir='tracking_results')
```

## Visualization
The system generates tracking videos with:
- Ground truth bounding boxes (green)
- Predicted bounding boxes (red)
- Search windows visualization
- Motion vector overlay
- IoU and frame information display

## Key References and Citations

1. SIFT (Scale-Invariant Feature Transform):
   > Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91-110.

2. ORB (Oriented FAST and Rotated BRIEF):
   > Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. 2011 International Conference on Computer Vision, 2564-2571.

3. HOG (Histogram of Oriented Gradients):
   > Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), 886-893.

4. LBP (Local Binary Patterns):
   > Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971-987.

5. Random Forests for Visual Tracking:
   > Gall, J., Yao, A., Razavi, N., Van Gool, L., & Lempitsky, V. (2011). Hough forests for object detection, tracking, and action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(11), 2188-2202.

6. Multi-Scale Sliding Windows:
   > Zhang, K., Zhang, L., & Yang, M. H. (2014). Fast compressive tracking. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(10), 2002-2015.

7. Camera Motion Compensation:
   > Wu, Y., Lim, J., & Yang, M. H. (2015). Object tracking benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9), 1834-1848.

## Future Improvements
1. Integration of deep learning features
2. Implementation of online learning mechanisms
3. Enhanced motion prediction models
4. Multi-object tracking support
5. Real-time performance optimizations
