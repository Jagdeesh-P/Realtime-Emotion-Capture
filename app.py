from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import os
import time
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow logging (moved to the beginning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Load emotion detection model
model_path = 'emotion_detection_model_fine_tuned.keras'
model = None  # Initialize model to None
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    model = load_model(model_path)
    logging.info("Emotion detection model loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model: {e}")
except Exception as e:
    logging.exception("An unexpected error occurred during model loading:")

additional_info = {
"developer": "Jagdeesh",
"project": "Real-Time Emotion Detection",
"description": "Using Advanced CNN Model to analyze emotions from facial expressions",
"version": "1.5",
"status": "Running",
}

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar cascade for face detection
face_cascade = None  # Initialize face_cascade to None
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logging.info("Haar cascade classifier loaded.")
except Exception as e:
    logging.error(f"Error loading Haar cascade: {e}")

# Global variables for statistics
fps = 60  # Start with an optimal FPS
face_count = 0
process_time = 0
current_emotion = "Initializing..."
resolution = "N/A"
emotion_history = []
TARGET_FPS = 60  # Set desired FPS
MIN_FRAME_DELAY = 1 / TARGET_FPS  # Minimum delay between frames (seconds)
last_frame_time = time.time()  # Initialize last frame processing time

def detect_emotion(frame):
    global fps, face_count, process_time, current_emotion, emotion_history

    if face_cascade is None or model is None:
        logging.warning("Face cascade or model not loaded. Skipping detection.")
        return "Error: Model or face cascade not loaded", 0, 0, ""

    start_time = time.time()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)
        emotion = None

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

            prediction = model.predict(roi_gray)
            emotion = emotions[np.argmax(prediction)]
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if emotion:
            current_emotion = emotion
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            emotion_history.insert(0, {'emotion': emotion, 'timestamp': timestamp})  # Add to history
            if len(emotion_history) > 5:  # Keep only the 5 most recent
                emotion_history.pop()
        else:
            current_emotion = "No Face Detected"

        process_time = int((time.time() - start_time) * 1000)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Error encoding frame to JPEG")
            return current_emotion, face_count, process_time, ""
        frame_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return current_emotion, face_count, process_time, frame_base64

    except Exception as e:
        logging.exception("Error during emotion detection:")
        return "Error processing frame", 0, 0, ""

@app.route('/')
def index():
    return render_template('index.html', info=additional_info)

@app.route('/data')
def data():
    return jsonify({
        'emotion': current_emotion,
        'fps': int(fps),
        'resolution': resolution,
        'faceCount': face_count,
        'processTime': process_time,
        'emotionHistory': emotion_history  # Send emotion history
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global fps, resolution, last_frame_time

    current_time = time.time()
    elapsed_time = current_time - last_frame_time

    if elapsed_time < MIN_FRAME_DELAY:
        time.sleep(MIN_FRAME_DELAY - elapsed_time)
        current_time = time.time()  # Update current_time after sleep
        elapsed_time = current_time - last_frame_time

    data = request.get_json()
    frame_data_base64 = data['frame']

    try:
        frame_data = base64.b64decode(frame_data_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("Error decoding image")
            return jsonify({'error': 'Failed to decode image'}), 400

        if resolution == "N/A":
            height, width = frame.shape[:2]
            resolution = f"{width}x{height}"
            logging.info(f"Frame resolution set to: {resolution}")

        start_time = time.time()
        emotion, face_count, process_time, frame_base64 = detect_emotion(frame)
        end_time = time.time()
        processing_time = end_time - start_time

        # Exponential Moving Average for FPS stabilization
        if processing_time > 0:
            new_fps = 1 / processing_time
            fps = 0.9 * fps + 0.9 * new_fps  # Smooth transition

        # Clamp FPS within range 50-60
        fps = max(50, min(fps, 60))

        last_frame_time = current_time
        return jsonify({'emotion': current_emotion, 'faceCount': face_count, 'processTime': process_time, 'fps': int(fps), 'frame': frame_base64})

    except Exception as e:
        logging.exception("Error processing frame:")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(500)
def internal_server_error(e):
    logging.exception("Internal Server Error:")
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
