import cv2
import dlib
import numpy as np
from imutils import face_utils
import time
import json
import os
from datetime import datetime
from collections import deque
from dataclasses import dataclass
import logging
from typing import Tuple, Optional, List
import multiprocessing as mp
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fatigue_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DriverState(Enum):
    ACTIVE = "Active"
    DROWSY = "Drowsy"
    SLEEPING = "SLEEPING !!!"
    YAWNING = "Yawning"
    NO_FACE = "No face detected"
    CALIBRATING = "Calibrating"

@dataclass
class Config:
    """Configuration parameters for fatigue detection"""
    # Detection thresholds
    EAR_THRESHOLD: float = 0.25
    MAR_THRESHOLD: float = 0.8
    DROWSY_EAR_THRESHOLD: float = 0.21
    
    # Frame settings
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS: int = 30
    FRAME_SKIP: int = 2
    
    # Time thresholds (in seconds)
    MAX_NO_FACE_TIME: float = 30.0
    CALIBRATION_TIME: float = 5.0
    
    # Counter thresholds
    EAR_CONSEC_FRAMES: int = 6
    YAWN_CONSEC_FRAMES: int = 4
    DROWSY_FRAME_THRESHOLD: int = 10
    
    # Directory settings
    DATA_DIR: str = "data"
    LOG_DIR: str = "logs"
    CALIBRATION_DIR: str = "calibration_data"
    
    # Logging settings
    ENABLE_LOGGING: bool = True

    @classmethod
    def load(cls, config_path: str = "config.json") -> 'Config':
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                return cls(**config_dict)
        return cls()

    def save(self, config_path: str = "config.json") -> None:
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

class FatigueMetrics:
    """Handles calculation and tracking of fatigue metrics"""
    def __init__(self, config: Config):
        self.config = config
        self.ear_history = deque(maxlen=30)  # 1 second of history at 30 FPS
        self.mar_history = deque(maxlen=30)
        self.baseline_ear = None
        self.baseline_mar = None
        
        # State tracking
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.drowsy_counter = 0
        self.current_state = DriverState.ACTIVE
        
        # Activity tracking
        self.recent_ears = deque(maxlen=90)  # 3 seconds of history
        self.recent_mars = deque(maxlen=90)

    def update_baselines(self, ear: float, mar: float) -> None:
        """Update baseline metrics during calibration"""
        # Skip invalid values
        if ear <= 0 or mar <= 0:
            return
            
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        
        # Update baselines once we have enough samples
        if len(self.ear_history) == self.ear_history.maxlen:
            # Remove outliers before calculating baseline
            ear_values = np.array(self.ear_history)
            mar_values = np.array(self.mar_history)
            
            # Calculate Q1, Q3, and IQR for both metrics
            ear_q1, ear_q3 = np.percentile(ear_values, [25, 75])
            mar_q1, mar_q3 = np.percentile(mar_values, [25, 75])
            ear_iqr = ear_q3 - ear_q1
            mar_iqr = mar_q3 - mar_q1
            
            # Filter out outliers
            ear_filtered = ear_values[
                (ear_values >= ear_q1 - 1.5 * ear_iqr) & 
                (ear_values <= ear_q3 + 1.5 * ear_iqr)
            ]
            mar_filtered = mar_values[
                (mar_values >= mar_q1 - 1.5 * mar_iqr) & 
                (mar_values <= mar_q3 + 1.5 * mar_iqr)
            ]
            
            # Update baselines using filtered data
            self.baseline_ear = np.mean(ear_filtered)
            self.baseline_mar = np.mean(mar_filtered)
            
            # Log the baseline values
            logger.info(f"Updated baselines - EAR: {self.baseline_ear:.3f}, MAR: {self.baseline_mar:.3f}")

    def update_state(self, ear: float, mar: float) -> DriverState:
        """Update driver state based on current metrics"""
        self.recent_ears.append(ear)
        self.recent_mars.append(mar)
        
        # Update normalized values if we have baselines
        if self.baseline_ear is not None and self.baseline_mar is not None:
            norm_ear = ear / self.baseline_ear
            norm_mar = mar / self.baseline_mar
        else:
            norm_ear = ear
            norm_mar = mar

        # Check for yawn
        if norm_mar > self.config.MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= self.config.YAWN_CONSEC_FRAMES:
                return DriverState.YAWNING
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)

        # Check for closed eyes (sleeping)
        if norm_ear < self.config.EAR_THRESHOLD:
            self.eye_closed_counter += 1
            if self.eye_closed_counter >= self.config.EAR_CONSEC_FRAMES:
                return DriverState.SLEEPING
        else:
            self.eye_closed_counter = max(0, self.eye_closed_counter - 1)

        # Check for drowsiness
        if norm_ear < self.config.DROWSY_EAR_THRESHOLD:
            self.drowsy_counter += 1
            if self.drowsy_counter >= self.config.DROWSY_FRAME_THRESHOLD:
                return DriverState.DROWSY
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 1)

        # Check if active
        if len(self.recent_ears) == self.recent_ears.maxlen:
            ear_std = np.std(self.recent_ears)
            mar_std = np.std(self.recent_mars)
            
            # Natural variation indicates active state
            if ear_std > 0.02 or mar_std > 0.1:
                return DriverState.ACTIVE
        
        # Return current state if no changes
        return self.current_state

    def get_normalized_metrics(self, ear: float, mar: float) -> Tuple[float, float]:
        """Get metrics normalized against baselines"""
        if self.baseline_ear is None or self.baseline_mar is None:
            return ear, mar
            
        norm_ear = ear / self.baseline_ear if self.baseline_ear > 0 else ear
        norm_mar = mar / self.baseline_mar if self.baseline_mar > 0 else mar
        return norm_ear, norm_mar

class FatigueDetector:
    def __init__(self, config_path: str = "config.json"):
        self.config = Config.load(config_path)
        self.create_required_directories()
        
        # Initialize components
        self.metrics = FatigueMetrics(self.config)
        
        # Initialize face detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # State variables
        self.calibrated = False
        self.frame_count = 0
        self.face_detected = False
        self.last_face_time = time.time()
        
        # Performance optimization
        self.process_pool = mp.Pool(processes=2)

    def create_required_directories(self) -> None:
        """Create all required directories for the application"""
        directories = [
            self.config.DATA_DIR,
            self.config.LOG_DIR,
            self.config.CALIBRATION_DIR
        ]
        
        for directory in directories:
            directory_path = Path(directory)
            if not directory_path.exists():
                directory_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")

    def calculate_ear(self, landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio"""
        # Extract eye coordinates
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Calculate EAR for each eye
        left_ear = self.compute_aspect_ratio(left_eye)
        right_ear = self.compute_aspect_ratio(right_eye)
        
        # Return average EAR
        return (left_ear + right_ear) / 2.0

    def calculate_mar(self, landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio using all mouth landmarks"""
        # Extract mouth landmarks
        mouth = landmarks[48:68]
        
        # Calculate vertical distances
        vert_dists = [
            np.linalg.norm(mouth[2] - mouth[10]),  # Upper lip to lower lip
            np.linalg.norm(mouth[3] - mouth[9]),   # Upper lip to lower lip
            np.linalg.norm(mouth[4] - mouth[8])    # Upper lip to lower lip
        ]
        
        # Calculate horizontal distance
        horiz_dist = np.linalg.norm(mouth[0] - mouth[6])  # Mouth corner to corner
        
        # Calculate MAR
        return np.mean(vert_dists) / horiz_dist if horiz_dist > 0 else 0

    @staticmethod
    def compute_aspect_ratio(landmarks: np.ndarray) -> float:
        """Compute aspect ratio for given landmarks"""
        vert_dists = [
            np.linalg.norm(landmarks[1] - landmarks[5]),
            np.linalg.norm(landmarks[2] - landmarks[4])
        ]
        horiz_dist = np.linalg.norm(landmarks[0] - landmarks[3])
        return sum(vert_dists) / (2.0 * horiz_dist) if horiz_dist > 0 else 0

    def handle_no_face(self) -> None:
        """Handle case when no face is detected"""
        self.face_detected = False
        if time.time() - self.last_face_time > self.config.MAX_NO_FACE_TIME:
            self.metrics.current_state = DriverState.NO_FACE
            if self.config.ENABLE_LOGGING:
                logger.warning("No face detected for extended period")

    def process_frame(self, frame: np.ndarray, calibration: bool = False) -> Optional[Tuple[float, float]]:
        """Process a single frame"""
        if self.frame_count % self.config.FRAME_SKIP != 0 and not calibration:
            self.frame_count += 1
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            self.handle_no_face()
            return None

        self.face_detected = True
        self.last_face_time = time.time()
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Calculate metrics
        ear = self.calculate_ear(landmarks)
        mar = self.calculate_mar(landmarks)

        if not calibration:
            new_state = self.metrics.update_state(ear, mar)
            self.metrics.current_state = new_state
            self.draw_info(frame, landmarks, ear, mar)

        return ear, mar

    def calibrate(self) -> None:
        """Calibrate the system for the current user"""
        logger.info("Starting calibration...")
        self.metrics.current_state = DriverState.CALIBRATING
        
        calibration_start = time.time()
        while time.time() - calibration_start < self.config.CALIBRATION_TIME:
            ret, frame = self.cap.read()
            if not ret:
                continue

            metrics = self.process_frame(frame, calibration=True)
            if metrics is not None:
                ear, mar = metrics
                self.metrics.update_baselines(ear, mar)

            # Show calibration progress
            progress = (time.time() - calibration_start) / self.config.CALIBRATION_TIME
            self.show_calibration_progress(frame, progress)
            cv2.imshow("Driver Fatigue Detection", frame)
            cv2.waitKey(1)

        self.calibrated = True
        self.save_calibration_data()
        logger.info("Calibration completed")

    def save_calibration_data(self) -> None:
        """Save calibration data"""
        calibration_data = {
            'baseline_ear': self.metrics.baseline_ear,
            'baseline_mar': self.metrics.baseline_mar,
            'timestamp': datetime.now().isoformat()
        }
        
        calibration_file = Path(self.config.CALIBRATION_DIR) / 'calibration.json'
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)

    def show_calibration_progress(self, frame: np.ndarray, progress: float) -> None:
        """Show calibration progress on frame"""
        bar_width = int(frame.shape[1] * 0.8)
        bar_height = 30
        bar_x = int((frame.shape[1] - bar_width) / 2)
        bar_y = int(frame.shape[0] * 0.9)

        # Draw progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (0, 255, 0), 2)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + int(bar_width * progress), bar_y + bar_height),
                     (0, 255, 0), -1)

    def draw_info(self, frame: np.ndarray, landmarks: np.ndarray, ear: float, mar: float) -> None:
        """Draw information on frame"""
        # Draw landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

        # Get status color based on state
        color = {
            DriverState.ACTIVE: (0, 255, 0),    # Green
            DriverState.DROWSY: (0, 255, 255),  # Yellow
            DriverState.SLEEPING: (0, 0, 255),  # Red
            DriverState.YAWNING: (255, 165, 0), # Orange
            DriverState.NO_FACE: (128, 128, 128) # Gray
        }.get(self.metrics.current_state, (0, 255, 0))

        # Draw status and metrics
        cv2.putText(frame, f"Status: {self.metrics.current_state.value}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.process_pool.close()
        self.process_pool.join()

    def run(self) -> None:
        """Main loop for fatigue detection"""
        try:
            # Perform initial calibration
            if not self.calibrated:
                self.calibrate()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break

                self.process_frame(frame)
                cv2.imshow("Driver Fatigue Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.calibrate()

        except KeyboardInterrupt:
            logger.info("Stopping detection...")
        finally:
            self.cleanup()

if __name__ == "__main__":
    detector = FatigueDetector()
    detector.run()
