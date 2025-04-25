# -*- coding: utf-8 -*-
"""
Calibration module for adapting the eye tracking system to individual users.
"""
import logging
import numpy as np
import time
import threading
import json
import os
from collections import deque

logger = logging.getLogger(__name__)

class CalibrationManager:
    """
    Manages the calibration process to adapt the system to individual users.
    """
    def __init__(self, blink_detector, calibration_file="calibration_data.json"):
        self.blink_detector = blink_detector
        self.calibration_file = calibration_file
        self.calibration_data = {}
        self.ear_values = deque(maxlen=500)  # Store recent EAR values for calibration
        self.calibration_in_progress = False
        self.calibration_step = 0
        self.calibration_start_time = 0
        self.lock = threading.Lock()
        
        # Set of calibration steps
        self.steps = [
            {"name": "relaxed", "duration": 5, "instruction": "Keep your eyes open naturally for 5 seconds"},
            {"name": "blink", "duration": 5, "instruction": "Blink normally 3-5 times over 5 seconds"},
            {"name": "closed", "duration": 3, "instruction": "Close your eyes for 3 seconds"},
            {"name": "fast_blink", "duration": 5, "instruction": "Blink quickly 5 times in 5 seconds"}
        ]
        
        # Try to load existing calibration data
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration data from file if it exists."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                logger.info(f"Loaded calibration data from {self.calibration_file}")
                
                # Apply calibration data to blink detector
                if "ear_threshold" in self.calibration_data:
                    self.blink_detector.set_calibrated_ear_threshold(
                        self.calibration_data["ear_threshold"])
            except Exception as e:
                logger.error(f"Failed to load calibration data: {str(e)}")
                self.calibration_data = {}

    def _save_calibration(self):
        """Save calibration data to file."""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f)
            logger.info(f"Saved calibration data to {self.calibration_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration data: {str(e)}")
            return False

    def start_calibration(self):
        """Start the calibration process."""
        with self.lock:
            if self.calibration_in_progress:
                logger.warning("Calibration already in progress")
                return False
            
            self.calibration_in_progress = True
            self.calibration_step = 0
            self.calibration_start_time = time.time()
            self.ear_values.clear()
            logger.info("Starting calibration process")
            return True

    def update_calibration(self, ear_value):
        """
        Update the calibration process with a new EAR value.
        Returns the current calibration status and instructions.
        """
        with self.lock:
            if not self.calibration_in_progress:
                return None
                
            # Store EAR value for analysis
            self.ear_values.append(ear_value)
            
            # Check if current step is complete
            current_time = time.time()
            elapsed_time = current_time - self.calibration_start_time
            
            if elapsed_time >= self.steps[self.calibration_step]["duration"]:
                # Move to next step or finish calibration
                self.calibration_step += 1
                self.calibration_start_time = current_time
                
                if self.calibration_step >= len(self.steps):
                    # Calibration complete, analyze data
                    self._analyze_calibration_data()
                    self.calibration_in_progress = False
                    return {"status": "completed", "message": "Calibration completed successfully!"}
            
            # Return current step information
            current_step = self.steps[self.calibration_step]
            remaining = current_step["duration"] - elapsed_time
            
            return {
                "status": "in_progress",
                "step": self.calibration_step + 1,
                "total_steps": len(self.steps),
                "instruction": current_step["instruction"],
                "remaining": max(0, round(remaining)),
                "progress": min(100, round((elapsed_time / current_step["duration"]) * 100))
            }

    def _analyze_calibration_data(self):
        """
        Analyze collected calibration data to determine optimal parameters.
        """
        if len(self.ear_values) < 50:
            logger.warning("Insufficient data for calibration analysis")
            return
            
        ear_array = np.array(list(self.ear_values))
        
        # Calculate statistics
        mean_ear = np.mean(ear_array)
        min_ear = np.min(ear_array)
        percentile_5 = np.percentile(ear_array, 5)
        
        # Determine optimal threshold - typically between the 5th percentile and minimum
        # This helps avoid false positives from minor fluctuations
        optimal_threshold = (percentile_5 + min_ear) / 2
        
        # Safety check - don't set threshold too high or low
        if optimal_threshold > 0.35:
            optimal_threshold = 0.35
        elif optimal_threshold < 0.15:
            optimal_threshold = 0.15
            
        logger.info(f"Calibration analysis complete. Optimal EAR threshold: {optimal_threshold:.3f}")
        
        # Update calibration data
        self.calibration_data["ear_threshold"] = optimal_threshold
        self.calibration_data["mean_ear"] = float(mean_ear)
        self.calibration_data["min_ear"] = float(min_ear)
        self.calibration_data["calibration_time"] = time.time()
        
        # Apply calibration to blink detector
        self.blink_detector.set_calibrated_ear_threshold(optimal_threshold)
        
        # Save calibration data
        self._save_calibration()

    def get_calibration_status(self):
        """Get the current calibration status."""
        with self.lock:
            if not self.calibration_in_progress:
                if self.calibration_data and "ear_threshold" in self.calibration_data:
                    return {"status": "calibrated", "threshold": self.calibration_data["ear_threshold"]}
                return {"status": "not_calibrated"}
                
            current_step = self.steps[self.calibration_step]
            elapsed_time = time.time() - self.calibration_start_time
            remaining = current_step["duration"] - elapsed_time
            
            return {
                "status": "in_progress",
                "step": self.calibration_step + 1,
                "total_steps": len(self.steps),
                "instruction": current_step["instruction"],
                "remaining": max(0, round(remaining)),
                "progress": min(100, round((elapsed_time / current_step["duration"]) * 100))
            }

    def cancel_calibration(self):
        """Cancel the current calibration process."""
        with self.lock:
            if not self.calibration_in_progress:
                return False
                
            self.calibration_in_progress = False
            logger.info("Calibration process cancelled")
            return True
