import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SlidingWindowTracker:
    def __init__(self, scale_factor=2.0, overlap=0.3):
        self.scale_factor = scale_factor
        self.overlap = overlap
        self.sift = cv2.SIFT_create()

        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def generate_search_windows(self, img_shape, prev_bbox):
        """Generate search windows around previous bounding box"""
        x, y, w, h = map(int, prev_bbox)

        # Calculate expanded window size
        window_w = int(w * self.scale_factor)
        window_h = int(h * self.scale_factor)

        # Calculate center position
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate step size based on overlap
        step_x = int(w * (1 - self.overlap))
        step_y = int(h * (1 - self.overlap))

        windows = []

        # Generate windows around the center
        for dy in range(-step_y, step_y + 1, max(1, step_y // 2)):
            for dx in range(-step_x, step_x + 1, max(1, step_x // 2)):
                win_x = max(0, min(center_x - window_w // 2 + dx, img_shape[1] - window_w))
                win_y = max(0, min(center_y - window_h // 2 + dy, img_shape[0] - window_h))

                windows.append((win_x, win_y, window_w, window_h))

        return windows

    def score_window(self, img, window, template, template_kp, template_desc):
        """Score how well a window matches the template"""
        x, y, w, h = map(int, window)
        roi = img[y:y+h, x:x+w]

        # Resize ROI to match template size
        roi = cv2.resize(roi, (template.shape[1], template.shape[0]))

        # Extract features
        kp, desc = self.sift.detectAndCompute(roi, None)

        if desc is None or template_desc is None or len(desc) == 0 or len(template_desc) == 0:
            return 0

        try:
            # Match features
            k = min(2, len(desc))
            matches = self.flann.knnMatch(template_desc, desc, k=k)

            # Apply ratio test
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                elif len(match) == 1:
                    good_matches.append(match[0])

            return len(good_matches)
        except Exception:
            return 0

class EnhancedVisualTrackingPipeline:
    def __init__(self, directoryPath, test_size=0.2, random_state=42):
        self.directoryPath = directoryPath
        self.sequencePath = directoryPath + '/sequences'
        self.annotationPath = directoryPath + '/annotations'
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
        self.feature_cache = {}

        # Initialize sliding window tracker
        self.window_tracker = SlidingWindowTracker()

        # Template tracking state
        self.template = None
        self.template_keypoints = None
        self.template_descriptors = None

    def extract_features_with_sliding_window(self, img, prev_bbox):
        """Enhanced feature extraction using sliding window approach"""
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = []

        if prev_bbox is not None:
            # Generate and evaluate search windows
            windows = self.window_tracker.generate_search_windows(img.shape, prev_bbox)

            if self.template is None:
                # Initialize template
                x, y, w, h = map(int, prev_bbox)
                self.template = gray[y:y+h, x:x+w].copy()
                self.template_keypoints, self.template_descriptors = self.window_tracker.sift.detectAndCompute(self.template, None)

            # Find best matching window
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

        # Resize ROI for consistent feature extraction
        roi = cv2.resize(roi, (64, 64))

        # Extract statistical features
        features.extend([
            np.mean(roi),
            np.std(roi),
            *np.percentile(roi, [25, 50, 75])
        ])

        # Extract edge features
        edges = cv2.Canny(roi, 100, 200)
        features.extend([
            np.mean(edges),
            np.std(edges)
        ])

        # Add position and size information
        if prev_bbox is not None:
            features.extend([x, y, w, h])
        else:
            features.extend([0, 0, 0, 0])

        # Add template matching score if available
        features.append(best_score if prev_bbox is not None else 0)

        return np.array(features)

    def process_sequence(self, sequence):
        """Process sequence with sliding window tracking"""
        ann_file = os.path.join(self.annotationPath, f"{sequence}.txt")
        annotations = np.loadtxt(ann_file, delimiter=',')
        img_files = sorted(glob(os.path.join(self.sequencePath, sequence, "*")))

        sequence_features = []
        sequence_labels = []
        sequence_paths = []

        prev_bbox = None
        template_update_counter = 0

        for img_path, bbox in zip(img_files, annotations):
            if img_path in self.feature_cache:
                features = self.feature_cache[img_path]
            else:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                features = self.extract_features_with_sliding_window(img, prev_bbox)
                self.feature_cache[img_path] = features

                # Update template periodically if tracking is stable
                template_update_counter += 1
                if template_update_counter >= 5 and prev_bbox is not None:  # More frequent updates
                    iou = self.calculate_iou(prev_bbox, bbox)
                    if iou > 0.6:  # More lenient threshold
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

    def prepare_data(self):
        """Prepare data with sequence splitting"""
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

    @staticmethod
    def calculate_iou(bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
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

    def create_tracking_video(self, sequence_data, y_pred, output_path, fps=30):
        """Create visualization video with search windows"""
        if len(sequence_data['paths']) == 0:
            return

        first_img = cv2.imread(sequence_data['paths'][0])
        if first_img is None:
            return

        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, (img_path, true_box, pred_box) in enumerate(zip(
            sequence_data['paths'],
            sequence_data['labels'],
            y_pred
        )):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Draw search windows
            if i > 0:
                windows = self.window_tracker.generate_search_windows(img.shape, y_pred[i-1])
                for window in windows:
                    x, y, w, h = map(int, window)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)

            # Draw true box (green)
            x, y, w, h = map(int, true_box)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, 'Ground Truth', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw predicted box (red)
            x, y, w, h = map(int, pred_box)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, 'Predicted', (x, y-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Calculate and display IoU
            iou = self.calculate_iou(true_box, pred_box)
            cv2.putText(img, f'IoU: {iou:.3f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(img, f'Frame: {i}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(img)

        out.release()

    def train_and_evaluate(self, output_dir='output_videos'):
        """Train model and evaluate results"""
        print("Preparing data...")
        X_train, y_train, train_data, test_data = self.prepare_data()

        X_train_scaled = self.scaler.fit_transform(X_train)

        print("Training model...")
        self.model.fit(X_train_scaled, y_train)

        os.makedirs(output_dir, exist_ok=True)

        print("Processing test sequences and creating videos...")
        all_test_predictions = []
        all_test_true = []

        for sequence in tqdm(test_data):
            X_test_scaled = self.scaler.transform(sequence['features'])
            y_pred = self.model.predict(X_test_scaled)

            video_path = os.path.join(output_dir, f"{sequence['sequence_name']}_tracking.mp4")
            self.create_tracking_video(sequence, y_pred, video_path)

            all_test_predictions.append(y_pred)
            all_test_true.append(sequence['labels'])

        y_test = np.vstack(all_test_true)
        y_pred = np.vstack(all_test_predictions)

        print("\nModel Evaluation Metrics:")
        coordinates = ['x', 'y', 'width', 'height']

        for i, coord in enumerate(coordinates):
            print(f"\nMetrics for {coord}:")
            print(f"MAE: {mean_absolute_error(y_test[:, i], y_pred[:, i]):.4f}")
            print(f"RMSE: {np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])):.4f}")
            print(f"R2 Score: {r2_score(y_test[:, i], y_pred[:, i]):.4f}")

        ious = [self.calculate_iou(true, pred) for true, pred in zip(y_test, y_pred)]
        print(f"\nMean IoU: {np.mean(ious):.4f}")

        print(f"\nTracking videos have been saved to {output_dir}/")

def main():
    """Main function to run the enhanced tracking pipeline"""
    directoryPath = './drive/MyDrive/MLP_MT24024/ObjectTracking'  # Update path as needed

    CONFIG = {
        'test_size': 0.2,
        'random_state': 42,
        'output_dir': 'enhanced_tracking_results'
    }

    try:
        print("Initializing Enhanced Visual Tracking Pipeline...")
        pipeline = EnhancedVisualTrackingPipeline(
            directoryPath=directoryPath,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state']
        )

        print("\nStarting training and evaluation process...")
        pipeline.train_and_evaluate(output_dir=CONFIG['output_dir'])

        print("\nEnhanced tracking pipeline completed successfully!")

    except Exception as e:
        print(f"\nError occurred during pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    