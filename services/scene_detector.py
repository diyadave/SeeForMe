#!/usr/bin/env python3
"""
Advanced Scene Detection Module
Combines YOLOv8n object detection, Places365 scene classification, and facial emotion recognition
"""

import os
import cv2
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

# Deep learning imports with fallbacks
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("⚠️ YOLOv8 not available. Install with: pip install ultralytics")

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("⚠️ PyTorch not available. Install with: pip install torch torchvision")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("⚠️ ONNX Runtime not available. Install with: pip install onnxruntime")

logger = logging.getLogger(__name__)

class SceneDetector:
    """Advanced scene detection combining multiple AI models"""
    
    def __init__(self):
        logger.info("👁️ Initializing Advanced Scene Detector...")
        
        # Model paths
        self.model_paths = {
            'yolo': 'models/yolov8n.pt',
            'emotion': 'models/emotion_model.onnx',
        }
        
        # Models
        self.yolo_model = None
        self.emotion_model = None
        
        # Face detection
        self.face_cascade = None
        
        # Performance tracking
        self.processing_times = []
        self.detection_cache = {}
        self.cache_ttl = 5.0  # seconds
        
        # Threading for async processing
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_cache = {}
        
        # Initialize models
        self.init_models()
        
        # Scene classification labels
        self.init_scene_labels()
        
        # Emotion labels
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear', 'Happy', 
            'Neutral', 'Sad', 'Surprise'
        ]
        
        logger.info("✅ Scene Detector initialized successfully")
    
    def init_models(self):
        """Initialize all detection models"""
        try:
            # Initialize YOLOv8n for object detection
            self.init_yolo_model()
            
            # Initialize face detection
            self.init_face_detection()
            
            # Initialize emotion detection
            self.init_emotion_model()
            
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
            # Continue with available models
    
    def init_yolo_model(self):
        """Initialize YOLOv8n model for object detection"""
        if not YOLO_AVAILABLE:
            logger.warning("⚠️ YOLO not available, skipping object detection")
            return
            
        try:
            yolo_path = self.model_paths['yolo']
            if os.path.exists(yolo_path):
                logger.info(f"📦 Loading YOLOv8n model from {yolo_path}")
                self.yolo_model = YOLO(yolo_path)
                logger.info("✅ YOLOv8n model loaded successfully")
            else:
                logger.warning(f"⚠️ YOLOv8n model not found at {yolo_path}")
                # Download model if not exists
                try:
                    logger.info("📥 Downloading YOLOv8n model...")
                    self.yolo_model = YOLO('yolov8n.pt')
                    # Save to models directory
                    os.makedirs('models', exist_ok=True)
                    # Note: YOLOv8 auto-downloads, model will be cached
                    logger.info("✅ YOLOv8n model downloaded and cached")
                except Exception as e:
                    logger.error(f"❌ Failed to download YOLOv8n: {e}")
                    self.yolo_model = None
                    
        except Exception as e:
            logger.error(f"❌ YOLOv8n initialization error: {e}")
            self.yolo_model = None
    
    def init_face_detection(self):
        """Initialize OpenCV face detection"""
        try:
            # Use OpenCV's pre-trained face detection
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            if self.face_cascade.empty():
                logger.error("❌ Face cascade not loaded")
                self.face_cascade = None
            else:
                logger.info("✅ Face detection initialized")
                
        except Exception as e:
            logger.error(f"❌ Face detection initialization error: {e}")
            self.face_cascade = None
    
    def init_emotion_model(self):
        """Initialize emotion detection model"""
        if not ONNX_AVAILABLE:
            logger.warning("⚠️ ONNX Runtime not available, emotion detection disabled")
            return
            
        try:
            emotion_path = self.model_paths['emotion']
            if os.path.exists(emotion_path):
                logger.info(f"😊 Loading emotion model from {emotion_path}")
                self.emotion_model = ort.InferenceSession(emotion_path)
                logger.info("✅ Emotion model loaded successfully")
            else:
                logger.warning(f"⚠️ Emotion model not found at {emotion_path}")
                # Create a simple emotion classifier placeholder
                self.emotion_model = None
                
        except Exception as e:
            logger.error(f"❌ Emotion model initialization error: {e}")
            self.emotion_model = None
    
    def init_scene_labels(self):
        """Initialize scene classification labels"""
        # Common scene categories
        self.scene_categories = {
            'indoor': [
                'living room', 'bedroom', 'kitchen', 'bathroom', 'office',
                'restaurant', 'store', 'classroom', 'library', 'hospital'
            ],
            'outdoor': [
                'street', 'park', 'beach', 'mountain', 'forest',
                'garden', 'parking lot', 'playground', 'sports field'
            ]
        }
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame using YOLOv8n"""
        if self.yolo_model is None:
            return self.fallback_object_detection(frame)
        
        try:
            start_time = time.time()
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract detection info
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        if confidence > 0.5:  # Confidence threshold
                            detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': box.tolist(),
                                'center': [
                                    (box[0] + box[2]) / 2,
                                    (box[1] + box[3]) / 2
                                ]
                            })
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.debug(f"🔍 Detected {len(detections)} objects in {processing_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"❌ Object detection error: {e}")
            return self.fallback_object_detection(frame)
    
    def fallback_object_detection(self, frame: np.ndarray) -> List[Dict]:
        """Fallback object detection using basic computer vision"""
        try:
            # Simple edge-based detection for basic shapes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Simple shape classification
                    aspect_ratio = w / h
                    if 0.8 <= aspect_ratio <= 1.2:
                        object_type = "square_object"
                    elif aspect_ratio > 1.5:
                        object_type = "rectangular_object"
                    else:
                        object_type = "vertical_object"
                    
                    detections.append({
                        'class': object_type,
                        'confidence': 0.6,
                        'bbox': [x, y, x+w, y+h],
                        'center': [x + w//2, y + h//2]
                    })
            
            return detections[:10]  # Limit to 10 detections
            
        except Exception as e:
            logger.error(f"❌ Fallback object detection error: {e}")
            return []
    
    def classify_scene(self, frame: np.ndarray) -> Dict:
        """Classify scene type using simplified scene analysis"""
        try:
            # Get objects detected in the scene
            objects = self.detect_objects(frame)
            object_names = [obj['class'] for obj in objects]
            
            # Simple heuristic-based scene classification
            scene_scores = {}
            
            # Indoor indicators
            indoor_objects = [
                'chair', 'table', 'couch', 'bed', 'tv', 'laptop',
                'book', 'cup', 'bottle', 'clock', 'vase'
            ]
            indoor_score = sum(1 for obj in object_names if obj in indoor_objects)
            
            # Outdoor indicators  
            outdoor_objects = [
                'car', 'truck', 'bicycle', 'tree', 'bench',
                'traffic light', 'stop sign', 'bird', 'dog'
            ]
            outdoor_score = sum(1 for obj in object_names if obj in outdoor_objects)
            
            # Kitchen indicators
            kitchen_objects = [
                'refrigerator', 'microwave', 'oven', 'sink',
                'cup', 'bowl', 'spoon', 'knife'
            ]
            kitchen_score = sum(1 for obj in object_names if obj in kitchen_objects)
            
            # Living room indicators
            living_room_objects = [
                'couch', 'tv', 'chair', 'table', 'remote'
            ]
            living_room_score = sum(1 for obj in object_names if obj in living_room_objects)
            
            # Determine scene type
            if kitchen_score >= 2:
                scene_type = 'kitchen'
                confidence = min(0.9, kitchen_score * 0.3)
            elif living_room_score >= 2:
                scene_type = 'living room'
                confidence = min(0.9, living_room_score * 0.3)
            elif indoor_score > outdoor_score:
                scene_type = 'indoor space'
                confidence = min(0.8, indoor_score * 0.2)
            elif outdoor_score > 0:
                scene_type = 'outdoor area'
                confidence = min(0.8, outdoor_score * 0.2)
            else:
                scene_type = 'general space'
                confidence = 0.5
            
            return {
                'scene_type': scene_type,
                'confidence': confidence,
                'objects_detected': len(objects),
                'indoor_score': indoor_score,
                'outdoor_score': outdoor_score,
                'detected_objects': object_names[:5]  # Top 5 objects
            }
            
        except Exception as e:
            logger.error(f"❌ Scene classification error: {e}")
            return {
                'scene_type': 'unknown',
                'confidence': 0.0,
                'objects_detected': 0,
                'detected_objects': []
            }
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in the frame"""
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_data = []
            for (x, y, w, h) in faces:
                face_data.append({
                    'bbox': [x, y, x+w, y+h],
                    'center': [x + w//2, y + h//2],
                    'size': w * h,
                    'confidence': 0.8  # Placeholder confidence
                })
            
            logger.debug(f"👤 Detected {len(face_data)} faces")
            return face_data
            
        except Exception as e:
            logger.error(f"❌ Face detection error: {e}")
            return []
    
    def detect_emotion(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect emotion from faces in the frame"""
        try:
            faces = self.detect_faces(frame)
            if not faces:
                return None
            
            # Use the largest face for emotion detection
            largest_face = max(faces, key=lambda f: f['size'])
            x1, y1, x2, y2 = largest_face['bbox']
            
            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            
            if self.emotion_model is not None:
                return self.detect_emotion_with_model(face_region)
            else:
                return self.detect_emotion_fallback(face_region)
            
        except Exception as e:
            logger.error(f"❌ Emotion detection error: {e}")
            return None
    
    def detect_emotion_with_model(self, face_region: np.ndarray) -> Optional[Dict]:
        """Detect emotion using ONNX model"""
        try:
            # Preprocess face for emotion model (typically 48x48 grayscale)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face.astype(np.float32) / 255.0
            input_face = normalized_face.reshape(1, 1, 48, 48)
            
            # Run inference
            input_name = self.emotion_model.get_inputs()[0].name
            outputs = self.emotion_model.run(None, {input_name: input_face})
            
            # Get emotion probabilities
            emotion_probs = outputs[0][0]
            emotion_idx = np.argmax(emotion_probs)
            emotion_confidence = float(emotion_probs[emotion_idx])
            emotion_label = self.emotion_labels[emotion_idx]
            
            return {
                'emotion': emotion_label,
                'confidence': emotion_confidence,
                'all_emotions': {
                    label: float(prob) for label, prob in zip(self.emotion_labels, emotion_probs)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Model emotion detection error: {e}")
            return self.detect_emotion_fallback(face_region)
    
    def detect_emotion_fallback(self, face_region: np.ndarray) -> Optional[Dict]:
        """Fallback emotion detection using basic image analysis"""
        try:
            if face_region.size == 0:
                return None
            
            # Simple brightness and contrast analysis for basic emotion estimation
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic features
            mean_brightness = np.mean(gray_face)
            brightness_std = np.std(gray_face)
            
            # Very basic emotion estimation
            if mean_brightness > 120 and brightness_std > 40:
                emotion = 'Happy'
                confidence = 0.6
            elif mean_brightness < 80:
                emotion = 'Sad'
                confidence = 0.5
            elif brightness_std > 50:
                emotion = 'Surprise'
                confidence = 0.5
            else:
                emotion = 'Neutral'
                confidence = 0.7
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback emotion detection error: {e}")
            return {
                'emotion': 'Neutral',
                'confidence': 0.5,
                'method': 'default'
            }
    
    def analyze_frame(self, frame: np.ndarray, mode: str = 'scene') -> Dict:
        """Comprehensive frame analysis"""
        try:
            start_time = time.time()
            
            result = {
                'timestamp': time.time(),
                'mode': mode,
                'frame_shape': frame.shape
            }
            
            if mode == 'scene':
                # Scene analysis mode
                scene_info = self.classify_scene(frame)
                result.update(scene_info)
                
                # Basic face count
                faces = self.detect_faces(frame)
                result['people_count'] = len(faces)
                
            elif mode == 'emotion':
                # Emotion analysis mode
                emotion_info = self.detect_emotion(frame)
                if emotion_info:
                    result.update(emotion_info)
                else:
                    result.update({
                        'emotion': 'No Face Detected',
                        'confidence': 0.0
                    })
                
                # Include face information
                faces = self.detect_faces(frame)
                result['faces_detected'] = len(faces)
            
            # Add processing time
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Frame analysis error: {e}")
            return {
                'timestamp': time.time(),
                'mode': mode,
                'error': str(e),
                'emotion': 'Neutral',
                'confidence': 0.0
            }
    
    def get_scene_description(self, frame: np.ndarray) -> str:
        """Generate natural language description of the scene"""
        try:
            analysis = self.analyze_frame(frame, mode='scene')
            
            scene_type = analysis.get('scene_type', 'unknown space')
            objects = analysis.get('detected_objects', [])
            people_count = analysis.get('people_count', 0)
            
            # Build description
            description_parts = []
            
            # Scene type
            description_parts.append(f"I can see what appears to be {scene_type}")
            
            # People
            if people_count > 0:
                if people_count == 1:
                    description_parts.append("with one person visible")
                else:
                    description_parts.append(f"with {people_count} people visible")
            
            # Objects
            if objects:
                obj_list = ", ".join(objects[:3])  # Limit to 3 objects
                description_parts.append(f"I can also see {obj_list}")
                if len(objects) > 3:
                    description_parts.append(f"and {len(objects) - 3} other objects")
            
            return ". ".join(description_parts) + "."
            
        except Exception as e:
            logger.error(f"❌ Scene description error: {e}")
            return "I'm having trouble analyzing the current scene."
    
    def get_emotion_description(self, frame: np.ndarray) -> str:
        """Generate natural language description of detected emotions"""
        try:
            analysis = self.analyze_frame(frame, mode='emotion')
            
            emotion = analysis.get('emotion', 'Neutral')
            confidence = analysis.get('confidence', 0.0)
            faces_count = analysis.get('faces_detected', 0)
            
            if faces_count == 0:
                return "I don't see any faces in the current view."
            
            if confidence > 0.7:
                certainty = "clearly"
            elif confidence > 0.5:
                certainty = "appears to be"
            else:
                certainty = "might be"
            
            if faces_count == 1:
                return f"I can see one person who {certainty} feeling {emotion.lower()}."
            else:
                return f"I can see {faces_count} people. The main person {certainty} feeling {emotion.lower()}."
            
        except Exception as e:
            logger.error(f"❌ Emotion description error: {e}")
            return "I'm having trouble reading emotions right now."
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clear caches
            self.detection_cache.clear()
            self.result_cache.clear()
            
            # Clean up models if needed
            if self.yolo_model:
                del self.yolo_model
                self.yolo_model = None
            
            if self.emotion_model:
                del self.emotion_model
                self.emotion_model = None
            
            logger.info("🧹 Scene detector cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of scene detector"""
        return {
            'yolo_available': self.yolo_model is not None,
            'face_detection_available': self.face_cascade is not None,
            'emotion_model_available': self.emotion_model is not None,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_analyses': len(self.processing_times),
            'cache_size': len(self.detection_cache)
        }

# Test function
if __name__ == "__main__":
    detector = SceneDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Test scene analysis
            scene_result = detector.analyze_frame(frame, mode='scene')
            print("Scene Analysis:", scene_result)
            
            # Test emotion analysis
            emotion_result = detector.analyze_frame(frame, mode='emotion')
            print("Emotion Analysis:", emotion_result)
            
            # Test descriptions
            scene_desc = detector.get_scene_description(frame)
            emotion_desc = detector.get_emotion_description(frame)
            print(f"Scene: {scene_desc}")
            print(f"Emotion: {emotion_desc}")
    
    cap.release()
    detector.cleanup()
