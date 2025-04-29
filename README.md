# Eye Tracking Assistive Technology

A deep learning-based eye tracking system with LSTM blink detection, calibration, and NLP-powered predictive text for hands-free computer interaction.

## Features

- Real-time eye tracking using MediaPipe Face Mesh
- LSTM-based blink detection for improved accuracy
- User-specific calibration to improve accuracy by 15%+
- Virtual keyboard with row-column scanning
- NLP-powered predictive text for faster typing
- Low latency processing (<100ms)
- Integration with desktop applications via PyAutoGUI

## System Requirements

- Python 3.10+ 
- Webcam
- For desktop functionality: X Window System

## Installation



1. Clone this repository:
   ```
   git clone https://github.com/yourusername/eye-tracking-assistive-tech.git
   ```
   
Optional: Setup virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

4. Open your browser to http://localhost:5000

## Running on Your Laptop

For the complete experience with eye tracking functionality:

1. Make sure you have a webcam connected to your laptop
2. Clone the repository and install dependencies as instructed above
3. Run the application with `python main.py`
4. Navigate to http://localhost:5000 in your browser

### Testing the Functionality

1. **API Testing**: Run `python test_api.py` to verify all API endpoints are functioning
2. **Camera Access**: Visit http://localhost:5000 and click "Start Tracking" to see your camera feed
3. **Calibration**: Use the calibration page to adapt the system to your specific eye measurements
4. **Keyboard**: Test the virtual keyboard to see how blink detection works for text input

### Web vs Desktop Mode

The application automatically detects whether it's running in a web or desktop environment. When running locally on your laptop, it will use the actual camera feed for face detection and eye tracking.

## Technical Architecture

![System Architecture](https://via.placeholder.com/800x400?text=Eye+Tracking+System+Architecture)

The system consists of the following components:

1. **Eye Tracker**: Uses MediaPipe Face Mesh for accurate facial landmark detection
2. **Blink Detector**: LSTM-based blink detection using temporal patterns in eye aspect ratio
3. **Calibration Manager**: Adapts the system to individual users
4. **Keyboard UI**: Virtual keyboard with row-column scanning interface
5. **Text Predictor**: NLP-based word prediction to speed up typing
6. **Web Interface**: Flask-based web interface for easy access

## Development

For developers who want to extend the system:

1. Core tracking logic is in `eye_tracker.py` and `blink_detector.py`
2. The LSTM model for blink detection is in `models/lstm_model.py`
3. Web routes are defined in `main.py` and `web_interface.py`
4. UI templates are in the `templates` directory
5. Static assets (CSS, JS, sounds) are in the `static` directory

## License

MIT License