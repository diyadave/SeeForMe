#!/usr/bin/env python3
"""
Emotion Detector - Facial emotion recognition using ONNX model
Detects emotions from face images for empathetic responses
"""

import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import os

logger = logging.getLogger(__name__)

class EmotionDetector:
    """Facial emotion detection using ONNX model"""
    
    def __init__(self):
        self.is_initialized = False
        self.onnx_session = None
        self.face_cascade = None
        
        # Emotion classes (FER2013 standard)
        self.emotion_classes = [
            'Angry', 'Disgust', 'Fear', 'Happy', 
            'Sad', 'Surprise', 'Neutral'
        ]
        
        # Model input size
        self.input_size = (48, 48)
        
        # Initialize components
        self.initialize_face_detection()
        self.initialize_emotion_model()
        
        logger.info("😊 Emotion detector initialized")
    
    def initialize_face_detection(self):
        """Initialize OpenCV face detection"""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error("❌ Failed to load face cascade")
                return False
            
            logger.info("✅ Face detection initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Face detection initialization failed: {e}")
            return False
    
    def initialize_emotion_model(self):
        """Initialize ONNX emotion recognition model"""
        try:
            import onnxruntime as ort
            
            # Look for emotion model in attached assets
            model_paths = [
                'attached_assets/emotion_model_1753972140817.onnx',
                'models/emotion_model.onnx',
                'emotion_model.onnx'
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                # Create ONNX runtime session
                self.onnx_session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                self.is_initialized = True
                logger.info(f"✅ Emotion model loaded: {model_path}")
            else:
                logger.warning("⚠️ Emotion model not found, using fallback detection")
                self.is_initialized = False
                
            return self.is_initialized
            
        except ImportError:
            logger.warning("⚠️ ONNX Runtime not available, using fallback detection")
            return False
        except Exception as e:
            logger.error(f"❌ Emotion model initialization failed: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [(x, y, w, h) for x, y, w, h in faces]
            
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return []
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion recognition"""
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_image
            
            # Resize to model input size
            face_resized = cv2.resize(face_gray, self.input_size)
            
            # Normalize pixel values
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions
            face_input = np.expand_dims(face_normalized, axis=0)  # Add batch dim
            face_input = np.expand_dims(face_input, axis=1)       # Add channel dim
            
            return face_input
            
        except Exception as e:
            logger.error(f"❌ Face preprocessing failed: {e}")
            return None
    
    def predict_emotion(self, face_input: np.ndarray) -> Optional[Dict[str, Any]]:
        """Predict emotion using ONNX model"""
        if not self.is_initialized or self.onnx_session is None:
            return self.fallback_emotion_detection()
        
        try:
            # Get input name
            input_name = self.onnx_session.get_inputs()[0].name
            
            # Run inference
            outputs = self.onnx_session.run(None, {input_name: face_input})
            predictions = outputs[0][0]  # Remove batch dimension
            
            # Get predicted emotion
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            emotion = self.emotion_classes[emotion_idx]
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': {
                    self.emotion_classes[i]: float(predictions[i]) 
                    for i in range(len(self.emotion_classes))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Emotion prediction failed: {e}")
            return self.fallback_emotion_detection()
    
    def fallback_emotion_detection(self) -> Dict[str, Any]:
        """Fallback emotion detection when ONNX model is not available"""
        # Simple fallback that returns neutral emotion
        return {
            'emotion': 'Neutral',
            'confidence': 0.7,
            'all_predictions': {emotion: 0.14 for emotion in self.emotion_classes},
            'fallback': True
        }
    
    def detect_emotion(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Main emotion detection function"""
        if image is None:
            return None
        
        try:
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                return {
                    'emotion': 'No face detected',
                    'confidence': 0.0,
                    'faces_count': 0
                }
            
            # Use the largest face (presumably the main subject)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_image = image[y:y+h, x:x+w]
            
            # Preprocess face
            face_input = self.preprocess_face(face_image)
            if face_input is None:
                return None
            
            # Predict emotion
            emotion_result = self.predict_emotion(face_input)
            
            if emotion_result:
                emotion_result['faces_count'] = len(faces)
                emotion_result['face_bbox'] = (x, y, w, h)
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"❌ Emotion detection failed: {e}")
            return None
    
    def detect_multiple_emotions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect emotions for multiple faces in image"""
        if image is None:
            return []
        
        try:
            # Detect all faces
            faces = self.detect_faces(image)
            emotions = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_image = image[y:y+h, x:x+w]
                
                # Preprocess face
                face_input = self.preprocess_face(face_image)
                if face_input is None:
                    continue
                
                # Predict emotion
                emotion_result = self.predict_emotion(face_input)
                
                if emotion_result:
                    emotion_result['face_id'] = i
                    emotion_result['face_bbox'] = (x, y, w, h)
                    emotions.append(emotion_result)
            
            return emotions
            
        except Exception as e:
            logger.error(f"❌ Multiple emotion detection failed: {e}")
            return []
    
    def analyze_emotion_trends(self, recent_emotions: List[str]) -> Dict[str, Any]:
        """Analyze emotion trends over time"""
        if not recent_emotions:
            return {'dominant_emotion': 'Neutral', 'stability': 'stable'}
        
        # Count emotion occurrences
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.keys(), key=emotion_counts.get)
        
        # Calculate stability (how consistent emotions are)
        total = len(recent_emotions)
        dominant_ratio = emotion_counts[dominant_emotion] / total
        
        if dominant_ratio > 0.7:
            stability = 'stable'
        elif dominant_ratio > 0.4:
            stability = 'varying'
        else:
            stability = 'unstable'
        
        return {
            'dominant_emotion': dominant_emotion,
            'stability': stability,
            'emotion_distribution': emotion_counts,
            'confidence': dominant_ratio
        }
    
    def get_emotion_description(self, emotion: str, confidence: float) -> str:
        """Get natural language description of emotion"""
        descriptions = {
            'Happy': 'You look happy and cheerful!',
            'Sad': 'You seem a bit sad. I\'m here if you need support.',
            'Angry': 'You appear frustrated or upset.',
            'Fear': 'You look worried or concerned about something.',
            'Surprise': 'You seem surprised by something!',
            'Disgust': 'You appear displeased or disgusted.',
            'Neutral': 'You have a calm, neutral expression.'
        }
        
        base_description = descriptions.get(emotion, f'You appear to be {emotion.lower()}.')
        
        if confidence < 0.5:
            return f"I think {base_description.lower()}"
        elif confidence > 0.8:
            return base_description
        else:
            return f"You seem to be {emotion.lower()}."
    
    def get_status(self) -> Dict[str, Any]:
        """Get current detector status"""
        return {
            'status': 'ready' if self.is_initialized else 'fallback',
            'face_detection': self.face_cascade is not None,
            'emotion_model': self.onnx_session is not None,
            'emotion_classes': self.emotion_classes,
            'input_size': self.input_size
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up emotion detector...")
        
        if self.onnx_session:
            self.onnx_session = None
        
        self.face_cascade = None
        self.is_initialized = False
        
        logger.info("✅ Emotion detector cleanup completed")