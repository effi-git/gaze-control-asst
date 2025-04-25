# -*- coding: utf-8 -*-
"""
Utility functions for the eye tracking system.
"""
import logging
import os
import time
import threading
import numpy as np

def setup_logging(level=logging.DEBUG):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Add file handler to log to file
    file_handler = logging.FileHandler(f'logs/eye_tracker_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return root_logger

class PerformanceTracker:
    """
    Track performance metrics of the system.
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.frame_times = []
        self.process_times = []
        self.detection_times = []
        self.lock = threading.Lock()
    
    def add_frame_time(self, time_ms):
        """Add a frame processing time measurement."""
        with self.lock:
            self.frame_times.append(time_ms)
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
    
    def add_process_time(self, time_ms):
        """Add a MediaPipe processing time measurement."""
        with self.lock:
            self.process_times.append(time_ms)
            if len(self.process_times) > self.window_size:
                self.process_times.pop(0)
    
    def add_detection_time(self, time_ms):
        """Add a detection (blink/gesture) time measurement."""
        with self.lock:
            self.detection_times.append(time_ms)
            if len(self.detection_times) > self.window_size:
                self.detection_times.pop(0)
    
    def get_stats(self):
        """Get performance statistics."""
        with self.lock:
            stats = {}
            
            if self.frame_times:
                stats['avg_frame_time'] = np.mean(self.frame_times)
                stats['max_frame_time'] = np.max(self.frame_times)
                stats['fps'] = 1000 / stats['avg_frame_time'] if stats['avg_frame_time'] > 0 else 0
            else:
                stats['avg_frame_time'] = 0
                stats['max_frame_time'] = 0
                stats['fps'] = 0
            
            if self.process_times:
                stats['avg_process_time'] = np.mean(self.process_times)
                stats['max_process_time'] = np.max(self.process_times)
            else:
                stats['avg_process_time'] = 0
                stats['max_process_time'] = 0
            
            if self.detection_times:
                stats['avg_detection_time'] = np.mean(self.detection_times)
                stats['max_detection_time'] = np.max(self.detection_times)
            else:
                stats['avg_detection_time'] = 0
                stats['max_detection_time'] = 0
            
            return stats

def create_data_directory():
    """Create necessary data directories."""
    dirs = ['data', 'logs', 'models']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
