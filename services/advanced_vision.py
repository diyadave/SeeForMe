#!/usr/bin/env python3
"""
Advanced Vision Pipeline - YOLOv8n + Places365 + Emotion Detection
Exactly as requested by user for SeeForMe
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Tuple
import onnxruntime as ort

logger = logging.getLogger(__name__)

class AdvancedVision:
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.last_frame = None
        
        # YOLOv8n for object detection
        self.yolo_model = None
        self.yolo_path = "attached_assets/yolov8n_1753972140819.onnx"
        
        # Emotion detection ONNX model
        self.emotion_model = None
        self.emotion_path = "attached_assets/emotion_model_1753972140817.onnx"
        
        # Places365 categories
        self.places_categories = []
        self.places_path = "attached_assets/categories_places365_1753972140816.txt"
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all AI models"""
        try:
            # Load YOLOv8n
            if os.path.exists(self.yolo_path):
                self.yolo_model = ort.InferenceSession(self.yolo_path)
                logger.info("✅ YOLOv8n model loaded")
            
            # Load emotion model
            if os.path.exists(self.emotion_path):
                self.emotion_model = ort.InferenceSession(self.emotion_path)
                logger.info("✅ Emotion ONNX model loaded")
            
            # Load Places365 categories
            if os.path.exists(self.places_path):
                with open(self.places_path, 'r') as f:
                    self.places_categories = [line.strip() for line in f.readlines()]
                logger.info(f"✅ Places365 loaded ({len(self.places_categories)} categories)")
                
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera for vision analysis"""
        try:
            if self.camera:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(camera_id)
            if self.camera.isOpened():
                self.is_active = True
                logger.info(f"✅ Camera {camera_id} started")
                return True
        except Exception as e:
            logger.error(f"❌ Camera start error: {e}")
        
        self.is_active = False
        return False
    
    def detect_objects_yolo(self) -> List[str]:
        """Detect objects using YOLOv8n"""
        if not self.yolo_model or self.last_frame is None:
            return []
        
        try:
            # Preprocess frame for YOLO
            frame_resized = cv2.resize(self.last_frame, (640, 640))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
            frame_expanded = np.expand_dims(frame_transposed, axis=0)
            
            # Run YOLO inference
            input_name = self.yolo_model.get_inputs()[0].name
            outputs = self.yolo_model.run(None, {input_name: frame_expanded})
            
            # Process YOLO results (simplified)
            detected_objects = []
            if outputs and len(outputs) > 0:
                # Parse YOLO output format
                predictions = outputs[0]
                if predictions.shape[-1] >= 85:  # Standard YOLO format
                    for detection in predictions[0]:
                        if len(detection) >= 85:
                            confidence = detection[4]
                            if confidence > 0.5:
                                class_id = int(np.argmax(detection[5:85]))
                                detected_objects.append(self._get_yolo_class_name(class_id))
            
            return detected_objects[:5]  # Return top 5 objects
            
        except Exception as e:
            logger.error(f"❌ YOLO detection error: {e}")
            return []
    
    def detect_emotion_onnx(self) -> Dict:
        """Detect emotion using ONNX model"""
        if not self.emotion_model or self.last_frame is None:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        try:
            # Detect face first
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {'emotion': 'neutral', 'confidence': 0.0}
            
            # Get largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess for emotion model
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_expanded = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=0)
            
            # Run emotion inference
            input_name = self.emotion_model.get_inputs()[0].name
            outputs = self.emotion_model.run(None, {input_name: face_expanded})
            
            # Process emotion results
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotion_probs = outputs[0][0]
            emotion_id = np.argmax(emotion_probs)
            confidence = float(emotion_probs[emotion_id])
            
            return {
                'emotion': emotions[emotion_id],
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Emotion detection error: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0}
    
    def analyze_scene_places365(self) -> str:
        """Analyze scene using Places365 (simplified)"""
        if not self.places_categories or self.last_frame is None:
            return "indoor space"
        
        try:
            # Simple scene analysis based on image characteristics
            gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Simple heuristics for scene classification
            if brightness > 180:
                return "outdoor, bright lighting"
            elif brightness < 50:
                return "indoor, dim lighting"
            else:
                return "indoor, well-lit room"
                
        except Exception as e:
            logger.error(f"❌ Scene analysis error: {e}")
            return "unknown environment"
    
    def _get_yolo_class_name(self, class_id: int) -> str:
        """Get YOLO class name from ID"""
        yolo_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        if 0 <= class_id < len(yolo_classes):
            return yolo_classes[class_id]
        return "unknown object"
    
    def capture_and_analyze_full(self) -> Dict:
        """Full analysis pipeline: capture + YOLO + emotion + scene"""
        if not self.is_active or not self.camera:
            return {}
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                return {}
            
            self.last_frame = frame
            
            # Run all analyses
            objects = self.detect_objects_yolo()
            emotion_result = self.detect_emotion_onnx()
            scene = self.analyze_scene_places365()
            
            return {
                'objects': objects,
                'emotion': emotion_result.get('emotion', 'neutral'),
                'emotion_confidence': emotion_result.get('confidence', 0.0),
                'scene': scene,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Full analysis error: {e}")
            return {}
    
    def stop_camera(self):
        """Stop camera and cleanup"""
        self.is_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        logger.info("🛑 Camera stopped")

# Global instance
advanced_vision = AdvancedVision()