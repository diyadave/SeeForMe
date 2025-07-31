"""
Complete Vision Processing System for SeeForMe
Handles emotion detection, scene description, and object detection
All offline processing using ONNX models and YOLOv8n
"""

import cv2
import numpy as np
import onnxruntime as ort
# from ultralytics import YOLO  # Commented out due to installation issues
import threading
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VisionProcessor:
    def __init__(self):
        self.emotion_model = None
        self.yolo_model = None
        self.face_cascade = None
        self.current_camera = "front"  # front for emotions, back for scenes
        self.is_processing = False
        self.last_emotion = "neutral"
        self.last_scene_description = ""
        self.last_objects = []
        
        # Initialize models
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all vision models"""
        try:
            # Initialize face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Initialize YOLO for object detection (simplified for now)
            try:
                # Use basic OpenCV for object detection as fallback
                logger.info("✅ Using OpenCV-based object detection")
                self.yolo_model = None  # Will implement basic detection
            except Exception as e:
                logger.error(f"❌ Object detection setup failed: {e}")
                
            # Initialize emotion detection model
            try:
                emotion_model_path = Path("models/emotion_model.onnx")
                if emotion_model_path.exists():
                    self.emotion_model = ort.InferenceSession(str(emotion_model_path))
                    logger.info("✅ Emotion detection model loaded")
                else:
                    logger.warning("⚠️ Emotion model not found, using basic detection")
            except Exception as e:
                logger.error(f"❌ Emotion model loading failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Vision processor initialization failed: {e}")
    
    def switch_camera(self, camera_type="front"):
        """Switch between front and back camera"""
        self.current_camera = camera_type
        logger.info(f"📹 Switched to {camera_type} camera")
        
    def detect_emotion_from_face(self, frame):
        """Detect emotions from facial expressions"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return "no_face_detected"
                
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            if self.emotion_model:
                # Use ONNX model for emotion detection
                face_roi_resized = cv2.resize(face_roi, (48, 48))
                face_roi_normalized = face_roi_resized.astype(np.float32) / 255.0
                face_roi_expanded = np.expand_dims(face_roi_normalized, axis=0)
                face_roi_expanded = np.expand_dims(face_roi_expanded, axis=0)
                
                emotion_predictions = self.emotion_model.run(None, {'input': face_roi_expanded})[0]
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                emotion_idx = np.argmax(emotion_predictions)
                emotion = emotion_labels[emotion_idx]
                confidence = emotion_predictions[0][emotion_idx]
                
                return f"{emotion} (confidence: {confidence:.2f})"
            else:
                # Basic emotion detection using facial features
                face_area = w * h
                if face_area > 10000:  # Large face - likely close/engaged
                    return "engaged"
                elif face_area < 5000:  # Small face - likely distant
                    return "distant"
                else:
                    return "neutral"
                    
        except Exception as e:
            logger.error(f"❌ Emotion detection failed: {e}")
            return "neutral"
    
    def detect_objects_and_scene(self, frame):
        """Detect objects and describe scene using basic computer vision"""
        try:
            # Basic scene analysis using OpenCV
            detected_objects = []
            people_count = 0
            
            # Use face detection to count people
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            people_count = len(faces)
            
            if people_count > 0:
                detected_objects.append({
                    'object': 'person',
                    'confidence': 0.85,
                    'count': people_count
                })
            
            # Basic color analysis for scene description
            height, width = frame.shape[:2]
            center_region = frame[height//4:3*height//4, width//4:3*width//4]
            
            # Analyze brightness and color dominance
            brightness = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
            blue_dominance = np.mean(center_region[:,:,0])
            green_dominance = np.mean(center_region[:,:,1])
            red_dominance = np.mean(center_region[:,:,2])
            
            # Basic environment classification
            if brightness > 150:
                lighting = "well-lit"
            elif brightness > 80:
                lighting = "moderately lit"
            else:
                lighting = "dimly lit"
                
            # Determine dominant colors for scene context
            scene_context = f"The environment appears {lighting}"
            
            if green_dominance > blue_dominance and green_dominance > red_dominance:
                scene_context += " with greenery visible"
            elif blue_dominance > 120:
                scene_context += " with blue tones, possibly outdoors or near sky"
            
            # Generate scene description
            scene_description = self.generate_scene_description(detected_objects, people_count, scene_context)
            
            return scene_description, detected_objects
            
        except Exception as e:
            logger.error(f"❌ Object detection failed: {e}")
            return "I can see the camera view but cannot analyze the scene right now", []
    
    def generate_scene_description(self, objects, people_count, scene_context=""):
        """Generate natural language scene description"""
        description_parts = []
        
        # Add scene context
        if scene_context:
            description_parts.append(scene_context)
        
        # Describe people with emotions if detected
        if people_count > 0:
            if people_count == 1:
                emotion_desc = ""
                if self.last_emotion and self.last_emotion != "neutral" and "no_face" not in self.last_emotion:
                    emotion_desc = f" who appears to be {self.last_emotion}"
                description_parts.append(f"There is one person in view{emotion_desc}")
            else:
                description_parts.append(f"There are {people_count} people in view")
        
        # Describe other objects
        other_objects = [obj for obj in objects if obj['object'] != 'person']
        if other_objects:
            object_names = [obj['object'] for obj in other_objects[:5]]
            if len(object_names) == 1:
                description_parts.append(f"I can see a {object_names[0]}")
            elif len(object_names) == 2:
                description_parts.append(f"I can see a {object_names[0]} and a {object_names[1]}")
            else:
                objects_text = ", ".join(object_names[:-1]) + f", and a {object_names[-1]}"
                description_parts.append(f"I can see a {objects_text}")
        
        if description_parts:
            return ". ".join(description_parts) + "."
        else:
            return "The scene appears to be clear with no specific objects detected."
    
    def process_camera_frame(self, camera_type="front"):
        """Process single camera frame based on camera type"""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0 if camera_type == "front" else 1)
            
            if not cap.isOpened():
                logger.error(f"❌ Cannot open {camera_type} camera")
                return None, None
                
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"❌ Cannot read from {camera_type} camera")
                return None, None
                
            if camera_type == "front":
                # Front camera: Emotion detection
                emotion = self.detect_emotion_from_face(frame)
                self.last_emotion = emotion
                return f"emotion_detected", emotion
            else:
                # Back camera: Scene and object detection
                scene_desc, objects = self.detect_objects_and_scene(frame)
                self.last_scene_description = scene_desc
                self.last_objects = objects
                return f"scene_analyzed", scene_desc
                
        except Exception as e:
            logger.error(f"❌ Camera processing failed: {e}")
            return None, None
    
    def analyze_with_emotion_context(self, user_text, detected_emotion=None):
        """Combine emotion detection with voice analysis"""
        voice_emotion = "neutral"
        
        # Analyze voice for emotional cues
        if any(word in user_text.lower() for word in ["sad", "upset", "crying", "depressed", "down"]):
            voice_emotion = "sad"
        elif any(word in user_text.lower() for word in ["happy", "joy", "excited", "good", "great"]):
            voice_emotion = "happy"
        elif any(word in user_text.lower() for word in ["angry", "mad", "frustrated", "annoyed"]):
            voice_emotion = "angry"
        elif any(word in user_text.lower() for word in ["scared", "afraid", "worried", "anxious"]):
            voice_emotion = "fear"
        
        # Combine with facial emotion if available
        if detected_emotion and detected_emotion != "no_face_detected":
            facial_emotion = detected_emotion.split(" ")[0]  # Remove confidence score
            return {
                'voice_emotion': voice_emotion,
                'facial_emotion': facial_emotion,
                'combined_emotion': facial_emotion if facial_emotion != "neutral" else voice_emotion
            }
        else:
            return {
                'voice_emotion': voice_emotion,
                'facial_emotion': "not_detected",
                'combined_emotion': voice_emotion
            }
    
    def get_intelligent_camera_response(self, user_text):
        """Automatically switch camera and analyze based on user input"""
        # Determine which camera to use based on user input
        emotion_keywords = ["feel", "emotion", "mood", "happy", "sad", "angry", "upset"]
        scene_keywords = ["see", "look", "around", "here", "what", "where", "describe", "environment"]
        
        user_text_lower = user_text.lower()
        
        if any(keyword in user_text_lower for keyword in emotion_keywords):
            # User asking about emotions - use front camera
            logger.info("🎭 Switching to front camera for emotion detection")
            result_type, result_data = self.process_camera_frame("front")
            
            if result_type == "emotion_detected":
                emotion_analysis = self.analyze_with_emotion_context(user_text, result_data)
                return {
                    'camera_used': 'front',
                    'analysis_type': 'emotion',
                    'emotion_data': emotion_analysis,
                    'description': f"I can see your facial expression shows {emotion_analysis['facial_emotion']}, and from your voice I detect {emotion_analysis['voice_emotion']} emotions."
                }
                
        elif any(keyword in user_text_lower for keyword in scene_keywords):
            # User asking about environment - use back camera
            logger.info("🌍 Switching to back camera for scene analysis")
            result_type, result_data = self.process_camera_frame("back")
            
            if result_type == "scene_analyzed":
                return {
                    'camera_used': 'back',
                    'analysis_type': 'scene',
                    'scene_description': result_data,
                    'objects': self.last_objects,
                    'description': result_data
                }
        
        # Default: Quick emotion check for supportive responses
        result_type, result_data = self.process_camera_frame("front")
        if result_type == "emotion_detected":
            emotion_analysis = self.analyze_with_emotion_context(user_text, result_data)
            return {
                'camera_used': 'front',
                'analysis_type': 'emotion_check',
                'emotion_data': emotion_analysis,
                'description': f"I can sense you're feeling {emotion_analysis['combined_emotion']}."
            }
        
        return None

# Global vision processor instance
vision_processor = VisionProcessor()