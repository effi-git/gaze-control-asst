# -*- coding: utf-8 -*-
"""
LSTM-based blink detector module for accurate blink detection
with reduced false positives compared to simple EAR threshold methods.
"""
import logging
import numpy as np
import tensorflow as tf
import time
import threading
import os
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)

class BlinkDetector:
    """
    Blink detector that uses LSTM neural network to classify blinks
    based on temporal patterns in the Eye Aspect Ratio (EAR) values.
    """
    def __init__(self, sequence_length=10, confidence_threshold=0.7, 
                 ear_threshold=0.2, model_path=None):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.ear_threshold = ear_threshold
        self.ear_history = deque(maxlen=sequence_length)
        self.blink_history = deque(maxlen=20)  # For storing recent blink times
        self.last_blink_time = 0
        self.consecutive_frames_below_threshold = 0
        self.consecutive_frames_threshold = 2
        self.lock = threading.Lock()
        self.calibrated_ear_threshold = ear_threshold
        
        # Fill history with default values
        self.ear_history.extend([0.5] * sequence_length)
        
        # Load or create LSTM model
        self.model = self._load_or_create_model(model_path)
        
        # Initialize TensorFlow Lite interpreter if model exists
        self.tflite_interpreter = None
        if model_path and os.path.exists(model_path + '.tflite'):
            self._initialize_tflite(model_path + '.tflite')

    def _initialize_tflite(self, tflite_path):
        """Initialize TensorFlow Lite interpreter for optimized inference."""
        try:
            self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.tflite_interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
            logger.info(f"TFLite model loaded successfully from {tflite_path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {str(e)}")
            self.tflite_interpreter = None

    def _load_or_create_model(self, model_path):
        """Load existing model or create a new one if not found."""
        if model_path and os.path.exists(model_path):
            try:
                logger.info(f"Loading LSTM model from {model_path}")
                return load_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
        
        logger.info("Creating new LSTM blink detection model")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def update_ear(self, ear_value):
        """
        Update the EAR history with new value and return whether a blink was detected.
        """
        with self.lock:
            self.ear_history.append(ear_value)
            
            # Simple threshold-based detection for comparison
            blink_detected_threshold = False
            if ear_value < self.calibrated_ear_threshold:
                self.consecutive_frames_below_threshold += 1
                if self.consecutive_frames_below_threshold >= self.consecutive_frames_threshold:
                    blink_detected_threshold = True
            else:
                self.consecutive_frames_below_threshold = 0
            
            # Use LSTM model for detection if available
            blink_detected_lstm = self.predict_blink()
            
            # Combine methods with priority to LSTM when available
            blink_detected = blink_detected_lstm if self.tflite_interpreter else blink_detected_threshold
            
            # Debounce to prevent multiple detections
            current_time = time.time()
            if blink_detected and (current_time - self.last_blink_time) > 0.5:
                self.last_blink_time = current_time
                self.blink_history.append(current_time)
                return True
            
            return False

    def predict_blink(self):
        """
        Use the LSTM model to predict if a blink is occurring based on recent EAR values.
        """
        if len(self.ear_history) < self.sequence_length:
            return False
        
        # Prepare input data for the model
        sequence = list(self.ear_history)
        sequence_array = np.array(sequence).reshape(1, self.sequence_length, 1)
        
        # Use TFLite if available (faster inference)
        if self.tflite_interpreter:
            try:
                self.tflite_interpreter.set_tensor(
                    self.input_details[0]['index'], 
                    sequence_array.astype(np.float32)
                )
                self.tflite_interpreter.invoke()
                prediction = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
                confidence = prediction[0][0]
            except Exception as e:
                logger.error(f"TFLite inference error: {str(e)}")
                confidence = 0
        else:
            # Use full model for inference
            confidence = self.model.predict(sequence_array, verbose=0)[0][0]
        
        return confidence > self.confidence_threshold
        
    def train_model(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Train the LSTM model with labeled data.
        X_train: numpy array with shape (n_samples, sequence_length, 1) containing EAR sequences
        y_train: numpy array with shape (n_samples, 1) containing blink labels (0 or 1)
        """
        logger.info(f"Training LSTM model with {len(X_train)} samples")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        logger.info("LSTM model training completed")
        return history

    def save_model(self, model_path):
        """Save the trained model to disk."""
        try:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Convert and save as TFLite model for faster inference
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            tflite_path = model_path + '.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"TFLite model saved to {tflite_path}")
            
            # Initialize TFLite interpreter
            self._initialize_tflite(tflite_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False

    def set_calibrated_ear_threshold(self, threshold):
        """Set the calibrated EAR threshold based on user calibration."""
        with self.lock:
            self.calibrated_ear_threshold = threshold
            logger.info(f"Calibrated EAR threshold set to {threshold}")

    def get_recent_blinks(self):
        """Get timestamps of recent blinks for analysis."""
        with self.lock:
            return list(self.blink_history)
