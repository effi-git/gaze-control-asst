# -*- coding: utf-8 -*-
"""
Web interface routes for the eye tracking system.
"""
import logging
import json
import time
import os
import threading
import base64
import cv2
import numpy as np
from flask import request, jsonify, Response, render_template, session

# Import Flask app without creating a circular import
import main 
app = main.app  # Use the app from main.py

# These will be accessed directly from main when needed, avoiding the circular import
# eye_tracker = None
# calibration_manager = None
# blink_detector = None

logger = logging.getLogger(__name__)

# Global variables for video streaming
latest_frame = None
frame_lock = threading.Lock()

def update_latest_frame(frame):
    """Update the latest frame for streaming."""
    global latest_frame
    with frame_lock:
        # Resize for streaming to reduce bandwidth
        resized = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_frame = buffer

@app.route('/api/calibration/start', methods=['POST'])
def start_calibration():
    """Start the calibration process."""
    if not hasattr(main, 'calibration_manager') or main.calibration_manager is None:
        # Return simulated response for web environment
        return jsonify({
            "status": "started",
            "web_simulation": True,
            "message": "Calibration simulation started in web environment"
        })
    
    try:
        if main.calibration_manager.start_calibration():
            return jsonify({"status": "started"})
        else:
            return jsonify({"status": "error", "message": "Calibration already in progress"}), 400
    except Exception as e:
        logger.error(f"Error starting calibration: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/calibration/status', methods=['GET'])
def get_calibration_status():
    """Get the current calibration status."""
    if not hasattr(main, 'calibration_manager') or main.calibration_manager is None:
        # Return a placeholder status for web environment
        placeholder_status = {
            "status": "web_simulation",
            "message": "Calibration simulation in web environment",
            "progress": 0,
            "ear_values": [],
            "completed": False,
            "instructions": "This is a simulation only. On a desktop system with a camera, you would see real-time calibration."
        }
        return jsonify(placeholder_status)
    
    try:
        status = main.calibration_manager.get_calibration_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting calibration status: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": "Error retrieving calibration status",
            "error": str(e)
        }), 500

@app.route('/api/calibration/cancel', methods=['POST'])
def cancel_calibration():
    """Cancel the calibration process."""
    if not hasattr(main, 'calibration_manager') or main.calibration_manager is None:
        # In web environment, return a simulated success response
        return jsonify({"status": "cancelled", "simulation": True})
    
    try:
        if main.calibration_manager.cancel_calibration():
            return jsonify({"status": "cancelled"})
        else:
            return jsonify({"status": "error", "message": "No calibration in progress"}), 400
    except Exception as e:
        logger.error(f"Error cancelling calibration: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current status of the system."""
    # Check if we're in a web environment
    is_web_env = os.environ.get('REPL_ID') or not os.environ.get('DISPLAY')
    
    # Check if the components are initialized
    has_eye_tracker = hasattr(main, 'eye_tracker') and main.eye_tracker is not None
    has_blink_detector = hasattr(main, 'blink_detector') and main.blink_detector is not None
    
    if is_web_env and (not has_eye_tracker or not has_blink_detector):
        # Return simulated status for web environment
        import random
        
        # Simulate a random EAR value that occasionally triggers blinks
        simulated_ear = 0.3 + random.random() * 0.1
        simulated_blink = random.random() < 0.05  # 5% chance of blink
        
        # Every 30 seconds or so, simulate a face detection loss
        face_detected = random.random() < 0.95  # 95% chance of face detection
        
        return jsonify({
            "running": main.running if hasattr(main, 'running') else False,
            "web_simulation": True,
            "face_detected": face_detected,
            "ear_value": simulated_ear,
            "blink_detected": simulated_blink,
            "performance": {
                "avg_frame_time": 30 + random.random() * 10,
                "fps": 25 + random.random() * 5,
                "avg_process_time": 20 + random.random() * 5
            }
        })
    
    # For non-web environment or initialized trackers
    try:
        status = {
            "running": hasattr(main, 'running') and main.running,
            "face_detected": False,
            "ear_value": 0.0,
            "blink_detected": False,
            "performance": {}
        }
        
        if has_eye_tracker:
            # Get actual status from eye tracker
            status["face_detected"] = getattr(main.eye_tracker, 'face_detected', False)
            status["ear_value"] = getattr(main.eye_tracker, 'last_ear_value', 0.0)
            
            # Get performance metrics
            frame_times = list(main.eye_tracker.frame_times) if hasattr(main.eye_tracker, 'frame_times') else []
            process_times = list(main.eye_tracker.process_times) if hasattr(main.eye_tracker, 'process_times') else []
            
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                status["performance"]["avg_frame_time"] = avg_frame_time
                status["performance"]["fps"] = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            
            if process_times:
                status["performance"]["avg_process_time"] = sum(process_times) / len(process_times)
        
        if has_blink_detector:
            # Get blink status
            try:
                recent_blinks = main.blink_detector.get_recent_blinks()
                if recent_blinks:
                    latest_blink = max(recent_blinks)
                    status["blink_detected"] = (time.time() - latest_blink) < 0.5
            except Exception as e:
                logger.warning(f"Error getting blink status: {str(e)}")
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving system status: {str(e)}",
            "running": False
        }), 500

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route - simplified for web environment.
    For Replit environment, we return a static placeholder instead of live streaming 
    to prevent worker timeouts.
    """
    # Check if we're in a web environment
    is_web_env = os.environ.get('REPL_ID') or not os.environ.get('DISPLAY')
    
    if is_web_env:
        # For web environment, return a static placeholder image
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw a gradient background
        for y in range(360):
            color_val = int(50 + (y / 360) * 100)
            cv2.line(placeholder, (0, y), (640, y), (color_val, color_val, color_val), 1)
            
        # Add informative text
        cv2.putText(placeholder, "Camera Feed Unavailable in Web Environment", 
                   (40, 100), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(placeholder, "This is a simulation environment only", 
                   (140, 150), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "Press 'Start Tracking' to begin demo", 
                   (140, 200), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add a border
        cv2.rectangle(placeholder, (5, 5), (635, 355), (100, 100, 100), 2)
        
        # Encode the image
        _, img_encoded = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 90])
        response = Response(img_encoded.tobytes(), mimetype='image/jpeg')
        
        logger.info("Serving static placeholder for video feed in web environment")
        return response
    else:
        # For desktop environment with actual camera, use streaming approach
        def generate_frames():
            global latest_frame
            try:
                while True:
                    with frame_lock:
                        if latest_frame is not None:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame.tobytes() + b'\r\n')
                        else:
                            # Create a simple waiting frame
                            empty_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                            cv2.putText(empty_frame, "Waiting for camera...", 
                                      (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            _, buffer = cv2.imencode('.jpg', empty_frame)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    # Use a shorter sleep time for actual streaming
                    time.sleep(0.03)
            except Exception as e:
                logger.error(f"Error generating video frames: {str(e)}")
        
        try:
            return Response(generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            logger.error(f"Failed to create video stream: {str(e)}")
            return "Video streaming unavailable", 500

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update system settings."""
    # Check if we're in a web environment
    is_web_env = os.environ.get('REPL_ID') or not os.environ.get('DISPLAY')
    
    if request.method == 'POST':
        try:
            settings = request.json
            
            # For web environment with no detectors initialized, return simulated success
            if is_web_env and (not blink_detector or not hasattr(app, 'keyboard_ui')):
                logger.info(f"Web simulation: settings update requested: {settings}")
                return jsonify({
                    "status": "success", 
                    "web_simulation": True,
                    "message": "Settings updated in simulation mode"
                })
            
            # Update blink detector settings if provided
            if blink_detector and 'ear_threshold' in settings:
                blink_detector.set_calibrated_ear_threshold(float(settings['ear_threshold']))
                logger.info(f"Updated EAR threshold to {settings['ear_threshold']}")
            
            # Update keyboard UI settings if provided
            if 'traversal_interval' in settings and hasattr(app, 'keyboard_ui') and app.keyboard_ui:
                app.keyboard_ui.traversal_interval = float(settings['traversal_interval'])
                logger.info(f"Updated traversal interval to {settings['traversal_interval']}")
            
            return jsonify({"status": "success"})
        
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 400
    
    else:  # GET
        # For web environment with no detectors initialized, return simulated settings
        if is_web_env and (not blink_detector or not hasattr(app, 'keyboard_ui')):
            return jsonify({
                "web_simulation": True,
                "ear_threshold": 0.25,  # Default EAR threshold
                "traversal_interval": 0.8,  # Default traversal speed
                "message": "Simulated settings for web environment"
            })
        
        settings = {}
        
        # Get blink detector settings
        if blink_detector:
            try:
                settings['ear_threshold'] = getattr(blink_detector, 'calibrated_ear_threshold', 0.25)
            except Exception as e:
                logger.warning(f"Error getting ear threshold: {str(e)}")
                settings['ear_threshold'] = 0.25  # Default
        
        # Get keyboard UI settings
        if hasattr(app, 'keyboard_ui') and app.keyboard_ui:
            try:
                settings['traversal_interval'] = getattr(app.keyboard_ui, 'traversal_interval', 0.8)
            except Exception as e:
                logger.warning(f"Error getting traversal interval: {str(e)}")
                settings['traversal_interval'] = 0.8  # Default
        
        return jsonify(settings)

# Additional route to handle errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html', error=str(e)), 500
