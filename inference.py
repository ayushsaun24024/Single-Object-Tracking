import os
import cv2
import joblib
import numpy as np
from pathlib import Path

class CameraMotionVisualizer:
    @staticmethod
    def draw_motion_grid(frame, transform_matrix, grid_size=32):
        if transform_matrix is None:
            return frame

        h, w = frame.shape[:2]
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                start = np.array([x, y, 1])
                end = np.dot(transform_matrix, start)
                if abs(end[0] - x) > 1 or abs(end[1] - y) > 1:
                    cv2.arrowedLine(
                        frame,
                        (int(x), int(y)),
                        (int(end[0]), int(end[1])),
                        (0, 255, 0),
                        1,
                        tipLength=0.2
                    )
        return frame


class SlidingWindowRefiner:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.scale_levels = 3
        self.scale_step = 1.2
        self.scale_factor = 2.0
        self.overlap = 0.3

        self.template = None
        self.template_kp = None
        self.template_desc = None

    def initialize_template(self, gray, bbox):
        x, y, w, h = map(int, bbox)
        self.template = gray[y:y+h, x:x+w].copy()
        self.template_kp, self.template_desc = self.sift.detectAndCompute(
            self.template, None
        )

    def generate_windows(self, img_shape, prev_bbox, transform_matrix=None):
        x, y, w, h = map(int, prev_bbox)

        if transform_matrix is not None:
            center = np.array([[x + w/2, y + h/2, 1]]).T
            transformed = np.dot(transform_matrix, center)
            x = int(transformed[0] - w/2)
            y = int(transformed[1] - h/2)

        windows = []
        for scale in np.linspace(1/self.scale_step, self.scale_step, self.scale_levels):
            ww = int(w * self.scale_factor * scale)
            hh = int(h * self.scale_factor * scale)

            step_x = int(ww * (1 - self.overlap))
            step_y = int(hh * (1 - self.overlap))

            cx, cy = x + w // 2, y + h // 2

            for dy in range(-step_y, step_y + 1, max(1, step_y // 2)):
                for dx in range(-step_x, step_x + 1, max(1, step_x // 2)):
                    wx = max(0, min(cx - ww // 2 + dx, img_shape[1] - ww))
                    wy = max(0, min(cy - hh // 2 + dy, img_shape[0] - hh))
                    windows.append((wx, wy, ww, hh))
        return windows

    def score_window(self, gray, window):
        if self.template_desc is None:
            return 0

        x, y, w, h = map(int, window)
        roi = gray[y:y+h, x:x+w]

        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return 0

        roi = cv2.resize(roi, self.template.shape[::-1])
        kp, desc = self.sift.detectAndCompute(roi, None)

        if desc is None or len(desc) < 2:
            return 0

        if len(self.template_desc) < 2:
            return 0

        matches = self.flann.knnMatch(self.template_desc, desc, k=2)
        
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if not good:
            return 0

        avg_dist = np.mean([m.distance for m in good])
        return len(good) * (1 - avg_dist / 512)


# ================================
# 🔹 ORIGINAL INFERENCE CLASS
# ================================

class ObjectTrackerInference:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        
        print("Loading pre-trained models...")
        self.position_model = joblib.load(os.path.join(model_dir, 'position_model.joblib'))
        self.size_model = joblib.load(os.path.join(model_dir, 'size_model.joblib'))
        self.position_scaler = joblib.load(os.path.join(model_dir, 'position_scaler.joblib'))
        self.size_scaler = joblib.load(os.path.join(model_dir, 'size_scaler.joblib'))
        print("Models loaded successfully!")
        
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None

        # 🔹 ADDITIVE
        self.window_refiner = SlidingWindowRefiner()
        self.template_initialized = False
        
    def estimate_camera_motion(self, frame):
        if frame is None:
            return np.eye(2, 3, dtype=np.float32)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_desc = desc
            return np.eye(2, 3, dtype=np.float32)
            
        if desc is None or self.prev_desc is None or len(desc) < 4 or len(self.prev_desc) < 4:
            return np.eye(2, 3, dtype=np.float32)
            
        matches = self.matcher.match(self.prev_desc, desc)
        
        if len(matches) < 4:
            return np.eye(2, 3, dtype=np.float32)
            
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(len(matches), 50)]
        
        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if transform_matrix is None:
            transform_matrix = np.eye(2, 3, dtype=np.float32)
            
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_desc = desc
        
        return transform_matrix
    
    def local_binary_pattern(self, image, n_points=8, radius=1):
        rows, cols = image.shape
        output = np.zeros((rows, cols))
        
        for i in range(radius, rows-radius):
            for j in range(radius, cols-radius):
                center = image[i, j]
                pattern = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    x1, x2 = int(np.floor(x)), int(np.ceil(x))
                    y1, y2 = int(np.floor(y)), int(np.ceil(y))
                    
                    f11 = image[y1, x1]
                    f12 = image[y1, x2]
                    f21 = image[y2, x1]
                    f22 = image[y2, x2]
                    
                    x_weight = x - x1
                    y_weight = y - y1
                    
                    pixel_value = (f11 * (1-x_weight) * (1-y_weight) +
                                 f21 * (1-x_weight) * y_weight +
                                 f12 * x_weight * (1-y_weight) +
                                 f22 * x_weight * y_weight)
                    
                    pattern |= (pixel_value > center) << k
                    
                output[i, j] = pattern
                
        return output
    
    def extract_features(self, frame, bbox, transform_matrix=None):
        if frame is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = map(int, bbox)
        
        x = max(0, min(x, gray.shape[1] - w))
        y = max(0, min(y, gray.shape[0] - h))
        w = min(w, gray.shape[1] - x)
        h = min(h, gray.shape[0] - y)
        
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            roi = gray
            
        roi = cv2.resize(roi, (64, 64))
        
        features = []
        
        hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
        hog_features = hog.compute(roi)
        features.extend(hog_features.flatten()[:64])
        
        lbp = self.local_binary_pattern(roi, n_points=8, radius=1)
        features.extend([
            np.mean(lbp),
            np.std(lbp),
            *np.percentile(lbp, [25, 50, 75])
        ])
        
        if transform_matrix is not None:
            features.extend([
                transform_matrix[0,0],
                transform_matrix[1,1],
                transform_matrix[0,2],
                transform_matrix[1,2]
            ])
        else:
            features.extend([1, 1, 0, 0])
            
        features.extend([x, y, w, h])
        
        return np.array(features).reshape(1, -1)
    
    def predict_bbox(self, features):
        features_position = self.position_scaler.transform(features)
        features_size = self.size_scaler.transform(features)
        
        position_pred = self.position_model.predict(features_position)
        size_pred = self.size_model.predict(features_size)
        
        bbox = np.hstack([position_pred, size_pred])[0]
        
        return bbox
    
    def track_video(self, video_path, initial_bbox, output_path='output_tracked.mp4', fps=30):
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height}, {total_frames} frames")  # 🔹 ADD THIS
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        
        current_bbox = initial_bbox
        frame_idx = 0
        
        print("Tracking object...")  # 🔹 ADD THIS
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            transform_matrix = self.estimate_camera_motion(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self.template_initialized:
                self.window_refiner.initialize_template(gray, current_bbox)
                self.template_initialized = True

            windows = self.window_refiner.generate_windows(
                frame.shape, current_bbox, transform_matrix
            )

            best_score = -1
            best_window = None

            for win in windows:
                score = self.window_refiner.score_window(gray, win)
                xw, yw, ww, hh = map(int, win)
                cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hh), (0, 255, 255), 1)
                if score > best_score:
                    best_score = score
                    best_window = win

            if best_window is not None:
                current_bbox = best_window

            features = self.extract_features(frame, current_bbox, transform_matrix)
            if features is not None:
                current_bbox = self.predict_bbox(features)

            frame = CameraMotionVisualizer.draw_motion_grid(frame, transform_matrix)

            x, y, w, h = map(int, current_bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:  # 🔹 ADD THIS
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"Tracking complete! Video saved to: {output_path}")  # 🔹 ADD THIS
        return output_path
