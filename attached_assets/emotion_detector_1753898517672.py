# emotion_detector.py
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque

class EmotionDetector:
    def __init__(self, model_path="emotion_model.onnx"):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_session = ort.InferenceSession(model_path)
        self.emotion_history = deque(maxlen=5)
        self.last_emotion = "Neutral"
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        emotions = []
        for (x, y, w, h) in faces:
            try:
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_processed = cv2.resize(face_roi, (48, 48)).astype(np.float32) / 255.0
                face_input = face_processed.reshape(1, 48, 48, 1)
                
                # Predict emotion
                input_name = self.emotion_session.get_inputs()[0].name
                outputs = self.emotion_session.run(None, {input_name: face_input})
                prediction = np.argmax(outputs[0])
                emotion = self.emotion_labels[prediction]
                confidence = np.max(outputs[0])
                
                # Update history
                self.emotion_history.append(emotion)
                
                # Get stable emotion
                if len(self.emotion_history) >= 3:
                    stable_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                    if confidence > 0.65:
                        self.last_emotion = stable_emotion
                
                emotions.append({
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'stable_emotion': self.last_emotion,
                    'bbox': (int(x), int(y), int(w), int(h))
                })
                
            except Exception as e:
                print(f"Emotion detection error: {e}")
                continue
                
        return emotions