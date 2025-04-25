# -*- coding: utf-8 -*-
"""
Main entry point for the eye tracking assistive technology application.
This system uses deep learning for accurate eye tracking and blink detection
to provide hands-free computer interaction.
"""
import logging
import os
import threading
import time
from flask import Flask, render_template

from eye_tracker import EyeTracker, IS_WEB_ENV
from blink_detector import BlinkDetector
from calibration import CalibrationManager
from keyboard_ui import KeyboardUI
from text_predictor import TextPredictor
from utils import setup_logging

# Setup logging and create necessary directories
setup_logging()
logger = logging.getLogger(__name__)

# Create necessary directories
from utils import create_data_directory
create_data_directory()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "eyetracker_default_secret")

# Global state variables
running = False
tracker_thread = None
eye_tracker = None
keyboard_ui = None
calibration_manager = None
text_predictor = None
blink_detector = None

def start_tracking():
    """Initialize and start the eye tracking system in a separate thread."""
    global running, tracker_thread, eye_tracker, keyboard_ui, text_predictor, blink_detector, calibration_manager
    
    if running:
        logger.warning("Tracking system is already running")
        return
    
    try:
        # Initialize components
        logger.info("Initializing eye tracking system...")
        text_predictor = TextPredictor()
        blink_detector = BlinkDetector()
        calibration_manager = CalibrationManager(blink_detector)
        eye_tracker = EyeTracker(blink_detector, calibration_manager)
        keyboard_ui = KeyboardUI(eye_tracker, text_predictor)
        
        # Start tracking thread
        running = True
        tracker_thread = threading.Thread(target=tracking_loop)
        tracker_thread.daemon = True
        tracker_thread.start()
        logger.info("Eye tracking system started successfully")
    except Exception as e:
        logger.error(f"Failed to start eye tracking system: {str(e)}")
        running = False

def stop_tracking():
    """Stop the eye tracking system."""
    global running, eye_tracker
    
    if not running:
        logger.warning("Tracking system is not running")
        return
    
    try:
        logger.info("Stopping eye tracking system...")
        running = False
        if eye_tracker:
            eye_tracker.release()
        if tracker_thread:
            tracker_thread.join(timeout=1.0)
        logger.info("Eye tracking system stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping eye tracking system: {str(e)}")

def tracking_loop():
    """Main tracking loop that runs in a separate thread."""
    global running, eye_tracker, keyboard_ui
    
    logger.info("Tracking loop started")
    
    # For web interface integration - need to define update_latest_frame here
    # to avoid circular import issues
    update_frame_func = None
    
    try:
        # Get reference to the update_latest_frame function after import
        from web_interface import update_latest_frame
        update_frame_func = update_latest_frame
        logger.info("Web interface frame update function successfully loaded")
    except Exception as e:
        logger.warning(f"Could not import update_latest_frame: {e}")
    
    try:
        while running:
            if eye_tracker and keyboard_ui:
                frame = eye_tracker.process_frame()
                if frame is not None:
                    # Update keyboard UI
                    keyboard_ui.update(frame)
                    
                    # Send frame to web interface if available
                    if update_frame_func:
                        try:
                            update_frame_func(frame)
                        except Exception as e:
                            logger.error(f"Error updating web interface frame: {e}")
            
            # Reduce sleep time for smoother animation in web environment
            time.sleep(0.01)  # Small sleep to prevent CPU overuse
    except Exception as e:
        logger.error(f"Error in tracking loop: {str(e)}")
        running = False
    finally:
        logger.info("Tracking loop ended")

# Flask routes
@app.route('/')
def index():
    """Render the main application interface."""
    return render_template('index.html')

@app.route('/calibration')
def calibration():
    """Render the calibration interface."""
    return render_template('calibration.html')

@app.route('/settings')
def settings():
    """Render the settings interface."""
    return render_template('settings.html')

@app.route('/api/start', methods=['POST'])
def api_start():
    """API endpoint to start the eye tracking system."""
    start_tracking()
    return {"status": "started"}

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """API endpoint to stop the eye tracking system."""
    stop_tracking()
    return {"status": "stopped"}

# Import web interface routes after defining app
import web_interface

if __name__ == '__main__':
    logger.info("Starting application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
