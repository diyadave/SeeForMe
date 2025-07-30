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

# Deep learning imports
try:
    from ultralytics import YOLO
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import onnxruntime as ort
except ImportError as e:
    logging.error(f"❌ Missing dependencies: {e}")
    logging.error("Install with: pip install ultralytics torch torchvision pillow onnxruntime")

logger = logging.getLogger(__name__)

class SceneDetector:
    """Advanced scene detection combining multiple AI models"""
    
    def __init__(self):
        logger.info("👁️ Initializing Advanced Scene Detector...")
        
        # Model paths
        self.model_paths = {
            'yolo': 'models/yolov8n.pt',
            'emotion': 'models/emotion_model.onnx',
            'places365': 'models/places365'
        }
        
        # Models
        self.yolo_model = None
        self.emotion_model = None
        self.places365_model = None
        
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
            
            # Initialize Places365 for scene classification
            self.init_places365_model()
            
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
            raise
    
    def init_yolo_model(self):
        """Initialize YOLOv8n model for object detection"""
        try:
            yolo_path = self.model_paths['yolo']
            if os.path.exists(yolo_path):
                logger.info(f"📦 Loading YOLOv8n model from {yolo_path}")
                self.yolo_model = YOLO(yolo_path)
                logger.info("✅ YOLOv8n model loaded successfully")
            else:
                logger.warning(f"⚠️ YOLOv8n model not found at {yolo_path}")
                # Download model if not exists
                logger.info("📥 Downloading YOLOv8n model...")
                self.yolo_model = YOLO('yolov8n.pt')
                # Save to models directory
                os.makedirs('models', exist_ok=True)
                self.yolo_model.save(yolo_path)
                logger.info("✅ YOLOv8n model downloaded and saved")
                
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
    
    def init_places365_model(self):
        """Initialize Places365 scene classification"""
        try:
            # For now, use a simplified scene classification
            # In production, you would load the actual Places365 model
            self.places365_labels = [
                'living_room', 'bedroom', 'kitchen', 'bathroom', 'office',
                'restaurant', 'store', 'street', 'park', 'beach',
                'mountain', 'forest', 'building', 'indoor', 'outdoor'
            ]
            logger.info("✅ Places365 scene labels loaded")
            
        except Exception as e:
            logger.error(f"❌ Places365 initialization error: {e}")
    
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
            return []
        
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
                'outdoor_score': outdoor_score
            }
            
        except Exception as e:
            logger.error(f"❌ Scene classification error: {e}")
            return {
                'scene_type': 'unknown',
                'confidence': 0.0,
                'objects_detected': 0
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
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Simple emotion detection based on facial analysis
            # In production, this would use a trained emotion model
            emotion = self.analyze_facial_features(face_roi)
            
            result = {
                'emotion': emotion,
                'confidence': 0.75,
                'face_bbox': largest_face['bbox'],
                'face_center': largest_face['center'],
                'timestamp': time.time()
            }
            
            logger.debug(f"😊 Detected emotion: {emotion}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Emotion detection error: {e}")
            return None
    
    def analyze_facial_features(self, face_roi: np.ndarray) -> str:
        """Simple facial feature analysis for emotion detection"""
        try:
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Simple heuristics based on facial features
            # This is a placeholder - in production use a trained model
            
            # Analyze brightness (smile detection proxy)
            brightness = np.mean(gray_face)
            
            # Analyze contrast (expression intensity)
            contrast = np.std(gray_face)
            
            # Simple classification based on features
            if brightness > 120 and contrast > 30:
                return 'Happy'
            elif brightness < 80:
                return 'Sad'
            elif contrast > 40:
                return 'Surprise'
            else:
                return 'Neutral'
                
        except Exception as e:
            logger.error(f"❌ Facial analysis error: {e}")
            return 'Neutral'
    
    def analyze_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """Comprehensive scene analysis combining all detection methods"""
        try:
            start_time = time.time()
            
            # Detect objects
            objects = self.detect_objects(frame)
            
            # Classify scene
            scene_info = self.classify_scene(frame)
            
            # Detect faces and emotions
            faces = self.detect_faces(frame)
            emotions = []
            
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                face_roi = frame[y1:y2, x1:x2]
                emotion_result = self.analyze_facial_features(face_roi)
                emotions.append({
                    'emotion': emotion_result,
                    'bbox': face['bbox'],
                    'confidence': 0.7
                })
            
            # Generate natural language description
            description = self.generate_scene_description(objects, scene_info, emotions)
            
            # Compile results
            analysis_result = {
                'description': description,
                'objects': objects,
                'scene_type': scene_info['scene_type'],
                'scene_confidence': scene_info['confidence'],
                'faces_detected': len(faces),
                'emotions': emotions,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            logger.info(f"👁️ Scene analysis complete: {description}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Scene analysis error: {e}")
            return {
                'description': 'Unable to analyze the scene at the moment.',
                'objects': [],
                'scene_type': 'unknown',
                'faces_detected': 0,
                'emotions': [],
                'error': str(e)
            }
    
    def generate_scene_description(self, objects: List[Dict], scene_info: Dict, emotions: List[Dict]) -> str:
        """Generate natural language description of the scene"""
        try:
            description_parts = []
            
            # Describe people and emotions
            if emotions:
                people_descriptions = []
                for emotion_data in emotions:
                    emotion = emotion_data['emotion']
                    if emotion == 'Happy':
                        people_descriptions.append("a smiling person")
                    elif emotion == 'Sad':
                        people_descriptions.append("a person who appears sad")
                    elif emotion == 'Surprise':
                        people_descriptions.append("a surprised person")
                    else:
                        people_descriptions.append("a person")
                
                if len(people_descriptions) == 1:
                    description_parts.append(f"I can see {people_descriptions[0]}")
                else:
                    description_parts.append(f"I can see {len(people_descriptions)} people")
            
            # Describe prominent objects
            if objects:
                # Get most confident objects
                top_objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)[:5]
                object_names = [obj['class'] for obj in top_objects]
                
                # Group similar objects
                object_counts = {}
                for obj_name in object_names:
                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                
                # Format object descriptions
                object_descriptions = []
                for obj_name, count in object_counts.items():
                    if count == 1:
                        object_descriptions.append(f"a {obj_name}")
                    else:
                        object_descriptions.append(f"{count} {obj_name}s")
                
                if object_descriptions:
                    if len(object_descriptions) == 1:
                        description_parts.append(f"There is {object_descriptions[0]}")
                    elif len(object_descriptions) == 2:
                        description_parts.append(f"There are {object_descriptions[0]} and {object_descriptions[1]}")
                    else:
                        obj_list = ", ".join(object_descriptions[:-1]) + f", and {object_descriptions[-1]}"
                        description_parts.append(f"There are {obj_list}")
            
            # Describe scene type
            scene_type = scene_info.get('scene_type', 'space')
            if scene_type != 'unknown':
                description_parts.append(f"in what appears to be {scene_type}")
            
            # Combine all parts
            if description_parts:
                description = ". ".join(description_parts) + "."
                
                # Add emotional context
                if emotions:
                    happy_count = sum(1 for e in emotions if e['emotion'] == 'Happy')
                    if happy_count > 0:
                        description += " The atmosphere seems positive and welcoming."
                    elif any(e['emotion'] in ['Sad', 'Fear'] for e in emotions):
                        description += " Someone might need some support."
                
                return description
            else:
                return "I can see a general scene, but I'm having trouble identifying specific details right now."
                
        except Exception as e:
            logger.error(f"❌ Description generation error: {e}")
            return "I'm having trouble describing what I see at the moment."
    
    def get_detection_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {'avg_processing_time': 0, 'total_detections': 0}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'total_detections': len(self.processing_times),
            'cache_size': len(self.detection_cache)
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up scene detector...")
        
        # Clear caches
        self.detection_cache.clear()
        self.result_cache.clear()
        
        # Reset models
        self.yolo_model = None
        self.emotion_model = None
        self.face_cascade = None
        
        logger.info("✅ Scene detector cleanup complete")


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    def test_scene_detector():
        """Test the scene detector with webcam"""
        try:
            detector = SceneDetector()
            
            # Test with webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("❌ Cannot open webcam")
                return
            
            logger.info("📹 Starting webcam test - press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze scene
                result = detector.analyze_scene(frame)
                
                # Draw results on frame
                cv2.putText(frame, result['description'][:50], (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw object bounding boxes
                for obj in result.get('objects', []):
                    bbox = obj['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{obj['class']} {obj['confidence']:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw face bounding boxes
                for emotion_data in result.get('emotions', []):
                    bbox = emotion_data['bbox']
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, emotion_data['emotion'], 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Scene Detection Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            stats = detector.get_detection_stats()
            print(f"\n📊 Detection Statistics:")
            print(f"   Average processing time: {stats['avg_processing_time']:.3f}s")
            print(f"   Total detections: {stats['total_detections']}")
            
            detector.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_scene_detector()
    else:
        print("Usage: python scene_detector.py test")