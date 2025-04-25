# -*- coding: utf-8 -*-
"""
Eye tracking module using MediaPipe Face Mesh for accurate landmark detection.
"""
import cv2
import mediapipe as mp
import logging
import time
import numpy as np
import threading
from collections import deque

logger = logging.getLogger(__name__)

class EyeTracker:
    """
    Eye tracker class using MediaPipe Face Mesh for landmark detection
    and tracking eye movements and blinks.
    """
    def __init__(self, blink_detector, calibration_manager, 
                 frame_width=1280, frame_height=720):
        self.blink_detector = blink_detector
        self.calibration_manager = calibration_manager
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # Camera setup
        self.cap = None
        self.initialize_camera()
        
        # Performance metrics
        self.frame_times = deque(maxlen=30)
        self.process_times = deque(maxlen=30)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Face detection status
        self.face_detected = False
        self.last_ear_value = 0.0
        self.last_blink_time = 0
        self.frame_count = 0
        
        logger.info("Eye tracker initialized")

    def initialize_camera(self):
        """Initialize the camera with specified resolution."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Cannot open webcam")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            logger.info(f"Camera initialized. Resolution: {int(actual_width)}x{int(actual_height)}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            return False

    def calculate_ear(self, eye_landmarks, frame_shape):
        """
        Calculate the Eye Aspect Ratio (EAR) for the given eye landmarks.
        """
        try:
            if not eye_landmarks or len(eye_landmarks) != 6:
                return 0.6
                
            # Convert normalized landmarks to pixel coordinates
            coords_list = []
            for lm in eye_landmarks:
                if (lm and hasattr(lm, 'x') and hasattr(lm, 'y') and 
                    np.isfinite(lm.x) and np.isfinite(lm.y)):
                    coords_list.append((lm.x * frame_shape[1], lm.y * frame_shape[0]))
                else:
                    coords_list.append((np.nan, np.nan))
            
            coords = np.array(coords_list)
            if np.isnan(coords).any():
                return 0.6
                
            # Calculate distances for EAR formula
            v1 = self.calculate_distance(coords[1], coords[5])
            v2 = self.calculate_distance(coords[2], coords[4])
            h1 = self.calculate_distance(coords[0], coords[3])
            
            if h1 <= 1e-6:
                return 0.6
                
            ear = (v1 + v2) / (2.0 * h1)
            return ear if np.isfinite(ear) else 0.6
        except Exception as e:
            logger.error(f"Error calculating EAR: {str(e)}")
            return 0.6

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        try:
            if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
                return 0.0
            if not (np.isfinite(p1[0]) and np.isfinite(p1[1]) and 
                    np.isfinite(p2[0]) and np.isfinite(p2[1])):
                return 0.0
                
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            sq_dist = dx**2 + dy**2
            
            return np.sqrt(sq_dist) if sq_dist >= 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return 0.0

    def process_frame(self):
        """
        Process a single frame from the camera:
        - Detect face and eye landmarks
        - Calculate EAR
        - Detect blinks
        - Update calibration if in progress
        
        Returns the processed frame with annotations.
        """
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                logger.error("Camera not initialized or closed")
                return None
                
            frame_start_time = time.time()
            
            try:
                success, frame = self.cap.read()
                if not success or frame is None:
                    logger.warning("Failed to read frame from camera")
                    return None
                    
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                frame_size = frame.shape
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                # Process with MediaPipe
                process_start = time.time()
                results = self.face_mesh.process(rgb_frame)
                process_end = time.time()
                
                # Record processing time
                self.process_times.append((process_end - process_start) * 1000)  # in ms
                
                # Reset face detection flag
                self.face_detected = False
                blink_detected = False
                ear_avg = 0.6  # Default EAR
                
                # Process face landmarks if detected
                if results.multi_face_landmarks:
                    self.face_detected = True
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate EAR for both eyes
                    left_eye_landmarks = [face_landmarks[i] for i in self.LEFT_EYE_INDICES]
                    right_eye_landmarks = [face_landmarks[i] for i in self.RIGHT_EYE_INDICES]
                    
                    left_ear = self.calculate_ear(left_eye_landmarks, frame_size)
                    right_ear = self.calculate_ear(right_eye_landmarks, frame_size)
                    ear_avg = (left_ear + right_ear) / 2.0
                    
                    # Update calibration if in progress
                    calibration_status = self.calibration_manager.update_calibration(ear_avg)
                    
                    # Detect blink using the blink detector
                    blink_detected = self.blink_detector.update_ear(ear_avg)
                    if blink_detected:
                        self.last_blink_time = time.time()
                        
                    # Draw landmarks for visualization
                    frame.flags.writeable = True
                    self._draw_eye_landmarks(frame, left_eye_landmarks, right_eye_landmarks)
                    
                    # Display calibration information if in progress
                    if calibration_status and calibration_status.get("status") == "in_progress":
                        self._draw_calibration_info(frame, calibration_status)
                
                # Store the last EAR value
                self.last_ear_value = ear_avg
                
                # Add performance metrics to frame
                self._add_performance_metrics(frame)
                
                # Add status indicators
                self._add_status_indicators(frame, blink_detected)
                
                # Calculate frame processing time
                frame_end_time = time.time()
                self.frame_times.append((frame_end_time - frame_start_time) * 1000)  # in ms
                
                self.frame_count += 1
                return frame
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                return None

    def _draw_eye_landmarks(self, frame, left_eye_landmarks, right_eye_landmarks):
        """Draw eye landmarks and EAR value on the frame."""
        try:
            # Draw landmarks for each eye
            for landmarks, is_left in [(left_eye_landmarks, True), (right_eye_landmarks, False)]:
                eye_label = "Left" if is_left else "Right"
                eye_color = (0, 255, 0)  # Green
                
                # Draw lines connecting eye landmarks
                points = []
                for lm in landmarks:
                    if hasattr(lm, 'x') and hasattr(lm, 'y'):
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 2, eye_color, -1)
                
                # Connect points to form eye outline
                if len(points) == 6:
                    for i in range(len(points)):
                        cv2.line(frame, points[i], points[(i + 1) % 6], eye_color, 1)
                
                # Calculate and display EAR for this eye
                ear = self.calculate_ear(landmarks, frame.shape)
                ear_text = f"{eye_label} EAR: {ear:.2f}"
                y_pos = 30 if is_left else 60
                cv2.putText(frame, ear_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, eye_color, 2)
        except Exception as e:
            logger.error(f"Error drawing eye landmarks: {str(e)}")

    def _draw_calibration_info(self, frame, calibration_status):
        """Draw calibration information on the frame."""
        try:
            step = calibration_status.get("step", 0)
            total_steps = calibration_status.get("total_steps", 0)
            instruction = calibration_status.get("instruction", "")
            remaining = calibration_status.get("remaining", 0)
            progress = calibration_status.get("progress", 0)
            
            # Draw calibration box
            cv2.rectangle(frame, (50, frame.shape[0] - 200), 
                          (frame.shape[1] - 50, frame.shape[0] - 50), (0, 0, 0), -1)
            cv2.rectangle(frame, (50, frame.shape[0] - 200), 
                          (frame.shape[1] - 50, frame.shape[0] - 50), (255, 255, 255), 2)
            
            # Draw step information
            step_text = f"Calibration Step {step}/{total_steps}"
            cv2.putText(frame, step_text, (100, frame.shape[0] - 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw instruction
            cv2.putText(frame, instruction, (100, frame.shape[0] - 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw time remaining
            time_text = f"Remaining: {remaining}s"
            cv2.putText(frame, time_text, (100, frame.shape[0] - 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw progress bar
            bar_width = frame.shape[1] - 200
            progress_width = int(bar_width * progress / 100)
            cv2.rectangle(frame, (100, frame.shape[0] - 70), 
                          (100 + bar_width, frame.shape[0] - 60), (100, 100, 100), -1)
            cv2.rectangle(frame, (100, frame.shape[0] - 70), 
                          (100 + progress_width, frame.shape[0] - 60), (0, 255, 0), -1)
        except Exception as e:
            logger.error(f"Error drawing calibration info: {str(e)}")

    def _add_performance_metrics(self, frame):
        """Add performance metrics to the frame."""
        try:
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
                
                # Display FPS
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display processing time
                if len(self.process_times) > 0:
                    avg_process_time = sum(self.process_times) / len(self.process_times)
                    process_text = f"Process: {avg_process_time:.1f}ms"
                    cv2.putText(frame, process_text, (frame.shape[1] - 230, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error adding performance metrics: {str(e)}")

    def _add_status_indicators(self, frame, blink_detected):
        """Add status indicators to the frame (face detection, blink detection)."""
        try:
            # Face detection status
            face_status = "Face: Detected" if self.face_detected else "Face: Not Detected"
            face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
            cv2.putText(frame, face_status, (10, frame.shape[0] - 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
            
            # Blink status
            blink_age = time.time() - self.last_blink_time
            if blink_detected:
                blink_text = "Blink: Detected"
                blink_color = (0, 255, 0)
            elif blink_age < 0.5:
                blink_text = "Blink: Recent"
                blink_color = (0, 255, 255)
            else:
                blink_text = "Blink: None"
                blink_color = (200, 200, 200)
                
            cv2.putText(frame, blink_text, (10, frame.shape[0] - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 2)
            
            # EAR value
            ear_text = f"EAR: {self.last_ear_value:.3f}"
            cv2.putText(frame, ear_text, (10, frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Error adding status indicators: {str(e)}")

    def release(self):
        """Release the camera and clean up resources."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            logger.info("Eye tracker resources released")
