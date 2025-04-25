/**
 * Main JavaScript file for Eye Tracking Assistive Technology
 * Provides common functionality used across all pages
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
    
    // Setup tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
});

/**
 * Updates the system status elements if they exist on the page
 * Common functionality used across all pages
 */
function updateSystemStatus() {
    fetch('/api/status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateStatusIndicators(data);
        })
        .catch(error => {
            console.error('Error fetching system status:', error);
            // Set offline status in case of error
            const statusIndicator = document.getElementById('statusIndicator');
            if (statusIndicator) {
                statusIndicator.className = 'badge rounded-pill bg-danger me-2';
                statusIndicator.innerHTML = '<i data-feather="alert-circle" class="me-1"></i> Error';
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
            }
        });
}

/**
 * Updates status indicators based on system data
 */
function updateStatusIndicators(data) {
    // Update global status indicator
    const statusIndicator = document.getElementById('statusIndicator');
    if (statusIndicator) {
        if (data.running) {
            statusIndicator.className = 'badge rounded-pill bg-success me-2';
            statusIndicator.innerHTML = '<i data-feather="activity" class="me-1"></i> Online';
        } else {
            statusIndicator.className = 'badge rounded-pill bg-secondary me-2';
            statusIndicator.innerHTML = '<i data-feather="activity" class="me-1"></i> Offline';
        }
    }
    
    // Update face detection indicator
    const faceIndicator = document.getElementById('faceIndicator');
    if (faceIndicator) {
        faceIndicator.textContent = data.face_detected ? 'Face Detected' : 'No Face';
        faceIndicator.className = data.face_detected ? 'badge bg-success' : 'badge bg-secondary';
    }
    
    // Update frame rate
    const frameRate = document.getElementById('frameRate');
    if (frameRate && data.performance && data.performance.fps) {
        frameRate.textContent = data.performance.fps.toFixed(1) + ' FPS';
    }
    
    // Update FPS indicator
    const fpsIndicator = document.getElementById('fpsIndicator');
    if (fpsIndicator && data.performance && data.performance.fps) {
        fpsIndicator.textContent = data.performance.fps.toFixed(1) + ' FPS';
    }
    
    // Update processing time
    const processTime = document.getElementById('processTime');
    if (processTime && data.performance && data.performance.avg_process_time) {
        processTime.textContent = data.performance.avg_process_time.toFixed(1) + ' ms';
    }
    
    // Update EAR value
    const earValue = document.getElementById('earValue');
    if (earValue) {
        earValue.textContent = data.ear_value.toFixed(2);
    }
    
    // Update blink detection status
    const blinkStatus = document.getElementById('blinkStatus');
    if (blinkStatus) {
        blinkStatus.textContent = data.blink_detected ? 'Detected' : 'None';
        blinkStatus.className = data.blink_detected ? 'text-success' : '';
    }
    
    // Refresh icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

/**
 * Starts the eye tracking system
 */
function startTracking() {
    fetch('/api/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'started') {
            // Hide overlay if it exists
            const overlay = document.getElementById('overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
            
            // Update buttons
            const startButton = document.getElementById('startButton');
            if (startButton) {
                startButton.disabled = true;
            }
            
            const stopButton = document.getElementById('stopButton');
            if (stopButton) {
                stopButton.disabled = false;
            }
            
            // Update status
            updateSystemStatus();
        }
    })
    .catch(error => {
        console.error('Error starting tracking:', error);
        alert('Failed to start eye tracking system: ' + error.message);
    });
}

/**
 * Stops the eye tracking system
 */
function stopTracking() {
    fetch('/api/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'stopped') {
            // Show overlay if it exists
            const overlay = document.getElementById('overlay');
            if (overlay) {
                overlay.style.display = 'flex';
            }
            
            // Update buttons
            const startButton = document.getElementById('startButton');
            if (startButton) {
                startButton.disabled = false;
            }
            
            const stopButton = document.getElementById('stopButton');
            if (stopButton) {
                stopButton.disabled = true;
            }
            
            // Update status
            updateSystemStatus();
        }
    })
    .catch(error => {
        console.error('Error stopping tracking:', error);
        alert('Failed to stop eye tracking system: ' + error.message);
    });
}

/**
 * Check calibration status
 */
function checkCalibrationStatus() {
    fetch('/api/calibration/status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const calibrationStatus = document.getElementById('calibrationStatus');
            if (calibrationStatus) {
                if (data.status === 'calibrated') {
                    calibrationStatus.textContent = 'Calibrated';
                    calibrationStatus.className = 'text-success';
                } else if (data.status === 'in_progress') {
                    calibrationStatus.textContent = 'In Progress';
                    calibrationStatus.className = 'text-warning';
                } else {
                    calibrationStatus.textContent = 'Not Calibrated';
                    calibrationStatus.className = '';
                }
            }
        })
        .catch(error => {
            console.error('Error checking calibration status:', error);
        });
}

// Set up automatic status updates if the page has status elements
if (document.getElementById('statusIndicator')) {
    // Update status immediately and then every second
    updateSystemStatus();
    setInterval(updateSystemStatus, 1000);
}

// Set up automatic calibration status updates if the page has calibration status elements
if (document.getElementById('calibrationStatus')) {
    // Check calibration status immediately and then every 2 seconds
    checkCalibrationStatus();
    setInterval(checkCalibrationStatus, 2000);
}
