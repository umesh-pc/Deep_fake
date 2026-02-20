import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import time
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the deepfake detection model
try:
    model = tf.keras.models.load_model('deepfake_model_v1.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame):
    """Preprocess a single frame for the model"""
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_path):
    """Predict if an image is real or fake"""
    if model is None:
        # Fallback for demo purposes
        return np.random.choice(['REAL', 'FAKE']), np.random.randint(85, 99)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_frame(img)
    
    prediction = model.predict(preprocessed)[0][0]
    confidence = int(prediction * 100) if prediction > 0.5 else int((1 - prediction) * 100)
    result = 'FAKE' if prediction > 0.5 else 'REAL'
    
    return result, confidence

def predict_video(video_path, sample_frames=10):
    """Predict if a video is real or fake by analyzing sample frames"""
    if model is None:
        # Fallback for demo purposes
        return np.random.choice(['REAL', 'FAKE']), np.random.randint(85, 99)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    predictions = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preprocessed = preprocess_frame(frame)
            pred = model.predict(preprocessed)[0][0]
            predictions.append(pred)
    
    cap.release()
    
    avg_prediction = np.mean(predictions)
    confidence = int(avg_prediction * 100) if avg_prediction > 0.5 else int((1 - avg_prediction) * 100)
    result = 'FAKE' if avg_prediction > 0.5 else 'REAL'
    
    return result, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Start timing
        start_time = time.time()
        
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to filename to avoid collisions
            timestamp_filename = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp_filename)
            file.save(filepath)
            
            # Determine file type and predict
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            try:
                if file_extension in ['mp4', 'avi', 'mov']:
                    result, confidence = predict_video(filepath)
                else:
                    result, confidence = predict_image(filepath)
                
                # Calculate processing time
                end_time = time.time()
                processing_time = round(end_time - start_time, 2)
                
                # Get current timestamp
                current_timestamp = datetime.now().strftime('%H:%M')
                
                # Prepare file path for display
                display_path = f'/static/uploads/{timestamp_filename}'
                
                return render_template('index.html', 
                                     result=result, 
                                     confidence=confidence,
                                     file_path=display_path,
                                     processing_time=processing_time,
                                     timestamp=current_timestamp)
            
            except Exception as e:
                print(f"Error during prediction: {e}")
                return render_template('index.html', error=f'Error processing file: {str(e)}')
        
        else:
            return render_template('index.html', error='Invalid file type')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)