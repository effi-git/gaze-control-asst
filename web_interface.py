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

# Initialize the running flag if it doesn't exist
if not hasattr(main, 'running'):
    main.running = False

# Import the IS_WEB_ENV variable to use consistently across all files
try:
    from eye_tracker import IS_WEB_ENV
except ImportError:
    # Fallback if not yet imported
    IS_WEB_ENV = os.environ.get('REPL_ID') is not None or not os.environ.get('DISPLAY')

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Handle OPTIONS requests for CORS preflight
@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    """Handle OPTIONS requests for CORS preflight."""
    return jsonify({})

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
    # Check if the components are initialized
    has_eye_tracker = hasattr(main, 'eye_tracker') and main.eye_tracker is not None
    has_blink_detector = hasattr(main, 'blink_detector') and main.blink_detector is not None
    
    if IS_WEB_ENV and (not has_eye_tracker or not has_blink_detector):
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
"""
@app.route('/video_feed')
def video_feed():

    
    from flask import send_file
    import io
    
    # In web environment, use a static image instead of streaming
    # to avoid worker timeouts
    if IS_WEB_ENV:
        try:
            # Create a simulated static image for web environment
            empty_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # Background color
            empty_frame[:, :] = (30, 30, 40)  # Dark blue-gray background
            
            # Draw a gradient background for visual interest
            for y in range(360):
                color_val = int(30 + (y / 360) * 20)
                cv2.line(empty_frame, (0, y), (640, y), (color_val, color_val, color_val+10), 1)
            
            # Draw simulated face outline
            face_center_x = 320
            face_center_y = 180
            
            # Face ellipse
            cv2.ellipse(empty_frame, 
                      (face_center_x, face_center_y), 
                      (100, 130), 0, 0, 360, (100, 150, 100), 2)
            
            # Eyes
            # Left eye
            eye_left_x = face_center_x - 40
            eye_left_y = face_center_y - 30
            
            # Right eye
            eye_right_x = face_center_x + 40
            eye_right_y = face_center_y - 30
            
            # Draw open eyes (circles)
            cv2.circle(empty_frame, (eye_left_x, eye_left_y), 15, (100, 200, 100), 2)
            cv2.circle(empty_frame, (eye_right_x, eye_right_y), 15, (100, 200, 100), 2)
            
            # Draw pupils
            cv2.circle(empty_frame, (eye_left_x, eye_left_y), 5, (150, 220, 150), -1)
            cv2.circle(empty_frame, (eye_right_x, eye_right_y), 5, (150, 220, 150), -1)
            
            # Draw mouth
            mouth_y = face_center_y + 40
            cv2.ellipse(empty_frame, 
                      (face_center_x, mouth_y), 
                      (30, 15), 0, 0, 180, (100, 180, 100), 2)
            
            # Add status text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(empty_frame, 'Face Detected', 
                      (20, 30), font, 0.7, (100, 200, 100), 2)
            
            # Show EAR value
            ear_value = 0.4
            cv2.putText(empty_frame, f'EAR: {ear_value:.2f}', 
                      (20, 60), font, 0.6, (180, 180, 180), 1)
            
            # Add environment info
            env_message = "Web Environment - Static Image Mode"
            cv2.putText(empty_frame, env_message, 
                      (20, 340), font, 0.5, (150, 150, 150), 1)
            
            # Add explanation
            cv2.putText(empty_frame, "API endpoints fully functional", 
                      (180, 310), font, 0.6, (200, 200, 200), 1)
            
            # Add a border
            cv2.rectangle(empty_frame, (5, 5), (635, 355), (100, 100, 100), 2)
            
            # Convert to JPEG image
            _, buffer = cv2.imencode('.jpg', empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Return as a file
            return send_file(
                io.BytesIO(buffer.tobytes()),
                mimetype='image/jpeg',
                as_attachment=False
            )
        except Exception as e:
            logger.error(f"Error generating static image: {str(e)}")
            logger.exception("Detailed static image error")
            return "Image generation error", 500
    else:
        # Only for desktop environments with real camera access
        def generate_frames():

            frame_count = 0
            
            try:
                while True:
                    with frame_lock:
                        if latest_frame is not None:
                            # Use the frame provided by the eye tracker
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + latest_frame.tobytes() + b'\r\n')
                        else:
                            # Create a simple waiting frame
                            empty_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(empty_frame, "Initializing camera...", 
                                      (220, 180), font, 0.7, (255, 255, 255), 2)
                            
                            _, buffer = cv2.imencode('.jpg', empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # Much shorter sleep - only for desktop environment
                    time.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error generating video frames: {str(e)}")
                logger.exception("Detailed frame generation error")
        
        try:
            return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            logger.error(f"Error returning video response: {str(e)}")
            return "Video streaming error", 500

            """

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update system settings."""
    # Check if the components are initialized
    has_blink_detector = hasattr(main, 'blink_detector') and main.blink_detector is not None
    has_keyboard_ui = hasattr(main, 'keyboard_ui') and main.keyboard_ui is not None
    
    if request.method == 'POST':
        try:
            settings = request.json
            
            # For web environment with no detectors initialized, return simulated success
            if IS_WEB_ENV and (not has_blink_detector or not has_keyboard_ui):
                logger.info(f"Web simulation: settings update requested: {settings}")
                return jsonify({
                    "status": "success", 
                    "web_simulation": True,
                    "message": "Settings updated in simulation mode"
                })
            
            # Update blink detector settings if provided
            if has_blink_detector and 'ear_threshold' in settings:
                main.blink_detector.set_calibrated_ear_threshold(float(settings['ear_threshold']))
                logger.info(f"Updated EAR threshold to {settings['ear_threshold']}")
            
            # Update keyboard UI settings if provided
            if has_keyboard_ui and 'traversal_interval' in settings:
                main.keyboard_ui.traversal_interval = float(settings['traversal_interval'])
                logger.info(f"Updated traversal interval to {settings['traversal_interval']}")
            
            return jsonify({"status": "success"})
        
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 400
    
    else:  # GET
        # For web environment with no detectors initialized, return simulated settings
        if IS_WEB_ENV and (not has_blink_detector or not has_keyboard_ui):
            return jsonify({
                "web_simulation": True,
                "ear_threshold": 0.25,  # Default EAR threshold
                "traversal_interval": 0.8,  # Default traversal speed
                "message": "Simulated settings for web environment"
            })
        
        settings = {}
        
        # Get blink detector settings
        if has_blink_detector:
            try:
                settings['ear_threshold'] = getattr(main.blink_detector, 'calibrated_ear_threshold', 0.25)
            except Exception as e:
                logger.warning(f"Error getting ear threshold: {str(e)}")
                settings['ear_threshold'] = 0.25  # Default
        
        # Get keyboard UI settings
        if has_keyboard_ui:
            try:
                settings['traversal_interval'] = getattr(main.keyboard_ui, 'traversal_interval', 0.8)
            except Exception as e:
                logger.warning(f"Error getting traversal interval: {str(e)}")
                settings['traversal_interval'] = 0.8  # Default
        
        return jsonify(settings)

# Additional route to handle errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Note: API start and stop endpoints are defined in main.py

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html', error=str(e)), 500
