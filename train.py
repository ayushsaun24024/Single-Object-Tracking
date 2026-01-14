import os
import cv2
import joblib
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class CameraMotionCompensator:
    def __init__(self):
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def estimate_motion(self, frame):
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

class ImprovedSlidingWindowTracker:
    def __init__(self, scale_factor=2.0, overlap=0.3):
        self.scale_factor = scale_factor
        self.overlap = overlap
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.scale_levels = 3
        self.scale_step = 1.2
        
    def generate_multiscale_windows(self, img_shape, prev_bbox, transform_matrix=None):
        x, y, w, h = map(int, prev_bbox)
        
        if transform_matrix is not None:
            center = np.array([[x + w/2, y + h/2, 1]], dtype=np.float32).T
            transformed_center = np.dot(transform_matrix, center)
            x = int(transformed_center[0] - w/2)
            y = int(transformed_center[1] - h/2)
        
        windows = []
        
        for scale in np.linspace(1/self.scale_step, self.scale_step, self.scale_levels):
            window_w = int(w * self.scale_factor * scale)
            window_h = int(h * self.scale_factor * scale)
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            step_x = int(window_w * (1 - self.overlap))
            step_y = int(window_h * (1 - self.overlap))
            
            for dy in range(-step_y, step_y + 1, max(1, step_y // 2)):
                for dx in range(-step_x, step_x + 1, max(1, step_x // 2)):
                    win_x = max(0, min(center_x - window_w // 2 + dx, img_shape[1] - window_w))
                    win_y = max(0, min(center_y - window_h // 2 + dy, img_shape[0] - window_h))
                    windows.append((win_x, win_y, window_w, window_h))
        
        return windows

    def score_window(self, img, window, template, template_kp, template_desc):
        x, y, w, h = map(int, window)
        roi = img[y:y+h, x:x+w]
        
        min_size = 20
        if roi.shape[0] < min_size or roi.shape[1] < min_size:
            return 0
            
        roi = cv2.resize(roi, (template.shape[1], template.shape[0]))
        
        kp, desc = self.sift.detectAndCompute(roi, None)
        
        if desc is None or template_desc is None or len(desc) == 0 or len(template_desc) == 0:
            return 0
            
        try:
            matches = self.flann.knnMatch(template_desc, desc, k=2)
            
            good_matches = []
            for match_group in matches:
                if len(match_group) == 2:
                    m, n = match_group
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                        
            if len(good_matches) == 0:
                return 0
                
            avg_distance = np.mean([m.distance for m in good_matches])
            score = len(good_matches) * (1 - avg_distance/512)
            
            return score
            
        except Exception:
            return 0

class ImprovedHybridTrackingPipeline:
    def __init__(self, directoryPath, test_size=0.2, random_state=42):
        self.directoryPath = directoryPath
        self.sequencePath = directoryPath + '/sequences'
        self.annotationPath = directoryPath + '/annotations'
        self.test_size = test_size
        self.random_state = random_state
        
        self.position_scaler = StandardScaler()
        self.size_scaler = StandardScaler()
        
        self.position_model = LinearRegression()
        self.size_model = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
        
        self.feature_cache = {}
        self.window_tracker = ImprovedSlidingWindowTracker()
        self.motion_compensator = CameraMotionCompensator()
        
        self.template = None
        self.template_keypoints = None
        self.template_descriptors = None
        
    def extract_enhanced_features(self, img, prev_bbox, transform_matrix=None):
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = []
        
        if prev_bbox is not None:
            windows = self.window_tracker.generate_multiscale_windows(
                img.shape, prev_bbox, transform_matrix
            )
            
            if self.template is None:
                x, y, w, h = map(int, prev_bbox)
                self.template = gray[y:y+h, x:x+w].copy()
                self.template_keypoints, self.template_descriptors = self.window_tracker.sift.detectAndCompute(self.template, None)
                
            best_score = -1
            best_window = None
            
            for window in windows:
                score = self.window_tracker.score_window(
                    gray, window, self.template,
                    self.template_keypoints, self.template_descriptors
                )
                
                if score > best_score:
                    best_score = score
                    best_window = window
                    
            if best_window is not None:
                x, y, w, h = best_window
                roi = gray[y:y+h, x:x+w]
            else:
                x, y, w, h = map(int, prev_bbox)
                roi = gray[y:y+h, x:x+w]
        else:
            x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]
            roi = gray
            
        roi = cv2.resize(roi, (64, 64))
        
        hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
        hog_features = hog.compute(roi)
        features.extend(hog_features.flatten()[:64])
        
        radius = 1
        n_points = 8 * radius
        lbp = self._local_binary_pattern(roi, n_points, radius)
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
        
        return np.array(features)
        
    def _local_binary_pattern(self, image, n_points, radius):
        rows = image.shape[0]
        cols = image.shape[1]
        output = np.zeros((rows, cols))
        for i in range(radius, rows-radius):
            for j in range(radius, cols-radius):
                center = image[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    x1 = int(np.floor(x))
                    x2 = int(np.ceil(x))
                    y1 = int(np.floor(y))
                    y2 = int(np.ceil(y))
                    
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
        
    def process_sequence(self, sequence):
        ann_file = os.path.join(self.annotationPath, f"{sequence}.txt")
        annotations = np.loadtxt(ann_file, delimiter=',')
        img_files = sorted(glob(os.path.join(self.sequencePath, sequence, "*")))
        
        sequence_features = []
        sequence_labels = []
        sequence_paths = []
        
        prev_bbox = None
        template_update_counter = 0
        prev_frame = None
        
        for img_path, bbox in zip(img_files, annotations):
            if img_path in self.feature_cache:
                features = self.feature_cache[img_path]
            else:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                transform_matrix = self.motion_compensator.estimate_motion(img)
                
                features = self.extract_enhanced_features(img, prev_bbox, transform_matrix)
                self.feature_cache[img_path] = features
                
                template_update_counter += 1
                if template_update_counter >= 5 and prev_bbox is not None:
                    iou = self.calculate_iou(prev_bbox, bbox)
                    if iou > 0.6:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        x, y, w, h = map(int, bbox)
                        self.template = gray[y:y+h, x:x+w].copy()
                        self.template_keypoints, self.template_descriptors = self.window_tracker.sift.detectAndCompute(self.template, None)
                        template_update_counter = 0
                        
            sequence_features.append(features)
            sequence_labels.append(bbox)
            sequence_paths.append(img_path)
            prev_bbox = bbox
            
        return {
            'features': np.array(sequence_features),
            'labels': np.array(sequence_labels),
            'paths': np.array(sequence_paths),
            'sequence_name': sequence
        }
        
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return max(0.0, min(1.0, iou))
        
    def prepare_data(self):
        sequence_folders = [f for f in os.listdir(self.sequencePath)
                          if os.path.isdir(os.path.join(self.sequencePath, f))]
                          
        train_sequences, test_sequences = train_test_split(
            sequence_folders,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print("Processing training sequences...")
        train_data = [self.process_sequence(seq) for seq in tqdm(train_sequences)]
        
        print("Processing test sequences...")
        test_data = [self.process_sequence(seq) for seq in tqdm(test_sequences)]
        
        X_train = np.vstack([seq['features'] for seq in train_data])
        y_train = np.vstack([seq['labels'] for seq in train_data])
        
        return X_train, y_train, train_data, test_data
        
    def create_tracking_video(self, sequence_data, y_pred, output_path, fps=30):
        if len(sequence_data['paths']) == 0:
            return
            
        first_img = cv2.imread(sequence_data['paths'][0])
        if first_img is None:
            return
            
        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        motion_tracker = self.motion_compensator
        
        for i, (img_path, true_box, pred_box) in enumerate(zip(
            sequence_data['paths'],
            sequence_data['labels'],
            y_pred
        )):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            transform_matrix = motion_tracker.estimate_motion(img)
            
            if i > 0:
                windows = self.window_tracker.generate_multiscale_windows(
                    img.shape, y_pred[i-1], transform_matrix
                )
                for window in windows:
                    x, y, w, h = map(int, window)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)
                    
            if i > 0:
                h, w = img.shape[:2]
                grid_size = 32
                for y in range(0, h, grid_size):
                    for x in range(0, w, grid_size):
                        start_point = np.array([x, y, 1])
                        end_point = np.dot(transform_matrix, start_point)
                        if np.abs(end_point[0] - x) > 1 or np.abs(end_point[1] - y) > 1:
                            cv2.arrowedLine(img, 
                                          (int(x), int(y)),
                                          (int(end_point[0]), int(end_point[1])),
                                          (0, 255, 0), 1, tipLength=0.2)
                                          
            x, y, w, h = map(int, true_box)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, 'Ground Truth', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
            x, y, w, h = map(int, pred_box)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, 'Prediction', (x, y-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                       
            iou = self.calculate_iou(true_box, pred_box)
            cv2.putText(img, f'IoU: {iou:.3f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                       
            cv2.putText(img, f'Frame: {i}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                       
            out.write(img)
            
        out.release()
        
    def train_and_evaluate(self, output_dir='output_videos'):
        print("Preparing data with enhanced features...")
        model_dir = os.path.join(output_dir, "models")
        X_train, y_train, train_data, test_data = self.prepare_data()
        
        X_train_position = self.position_scaler.fit_transform(X_train)
        X_train_size = self.size_scaler.fit_transform(X_train)
        
        y_train_position = y_train[:, :2]
        y_train_size = y_train[:, 2:]
        
        print("Training improved hybrid models...")
        self.position_model.fit(X_train_position, y_train_position)
        
        self.size_model.fit(X_train_size, y_train_size)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        print("Processing test sequences with motion compensation...")
        all_predictions = []
        all_test_true = []
        
        for sequence in tqdm(test_data):
            X_test_position = self.position_scaler.transform(sequence['features'])
            X_test_size = self.size_scaler.transform(sequence['features'])
            
            position_pred = self.position_model.predict(X_test_position)
            size_pred = self.size_model.predict(X_test_size)
            
            y_pred = np.hstack([position_pred, size_pred])
            
            video_path = os.path.join(output_dir, f"{sequence['sequence_name']}_tracking.mp4")
            self.create_tracking_video(sequence, y_pred, video_path)
            
            all_predictions.append(y_pred)
            all_test_true.append(sequence['labels'])
            
        y_test = np.vstack(all_test_true)
        y_pred = np.vstack(all_predictions)
        
        print("\nEnhanced Model Evaluation Metrics:")
        
        print("\nPosition Prediction Metrics:")
        for i, coord in enumerate(['x', 'y']):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            print(f"\n{coord.upper()} Coordinate:")
            print(f"MAE: {mae:.4f} pixels")
            print(f"RMSE: {rmse:.4f} pixels")
            print(f"R² Score: {r2:.4f}")
            
        print("\nSize Prediction Metrics:")
        for i, dim in enumerate(['width', 'height']):
            mae = mean_absolute_error(y_test[:, i+2], y_pred[:, i+2])
            rmse = np.sqrt(mean_squared_error(y_test[:, i+2], y_pred[:, i+2]))
            r2 = r2_score(y_test[:, i+2], y_pred[:, i+2])
            print(f"\n{dim.title()}:")
            print(f"MAE: {mae:.4f} pixels")
            print(f"RMSE: {rmse:.4f} pixels")
            print(f"R² Score: {r2:.4f}")
            
        ious = [self.calculate_iou(true, pred) for true, pred in zip(y_test, y_pred)]
        mean_iou = np.mean(ious)
        print(f"\nMean IoU: {mean_iou:.4f}")
        
        print(f"\nTracking videos saved to {output_dir}/")

        joblib.dump(self.position_model,
            os.path.join(model_dir, "position_model.joblib"))

        joblib.dump(self.size_model,
                    os.path.join(model_dir, "size_model.joblib"))
        
        joblib.dump(self.position_scaler,
                    os.path.join(model_dir, "position_scaler.joblib"))
        
        joblib.dump(self.size_scaler,
                    os.path.join(model_dir, "size_scaler.joblib"))
        
        print(f"Saved models and scalers to: {model_dir}")
        
        return {
            'position_metrics': {
                'x_mae': mean_absolute_error(y_test[:, 0], y_pred[:, 0]),
                'y_mae': mean_absolute_error(y_test[:, 1], y_pred[:, 1])
            },
            'size_metrics': {
                'width_mae': mean_absolute_error(y_test[:, 2], y_pred[:, 2]),
                'height_mae': mean_absolute_error(y_test[:, 3], y_pred[:, 3])
            },
            'mean_iou': mean_iou
        }

def main():
    directoryPath = "/kaggle/input/object-tracking-ml-dataset"
    
    CONFIG = {
        'test_size': 0.2,
        'random_state': 42,
        'output_dir': '/kaggle/working/improved_tracking_results'
    }
    
    try:
        print("Initializing Improved Hybrid Visual Tracking Pipeline...")
        pipeline = ImprovedHybridTrackingPipeline(
            directoryPath=directoryPath,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state']
        )
        
        print("\nStarting training and evaluation process...")
        results = pipeline.train_and_evaluate(output_dir=CONFIG['output_dir'])
        
        print("\nSummary of Results:")
        print("Position Prediction:")
        print(f"X coordinate MAE: {results['position_metrics']['x_mae']:.4f} pixels")
        print(f"Y coordinate MAE: {results['position_metrics']['y_mae']:.4f} pixels")
        print("\nSize Prediction:")
        print(f"Width MAE: {results['size_metrics']['width_mae']:.4f} pixels")
        print(f"Height MAE: {results['size_metrics']['height_mae']:.4f} pixels")
        print(f"\nOverall Mean IoU: {results['mean_iou']:.4f}")
        
        print("\nImproved tracking pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
