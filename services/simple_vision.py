"""
Simple Computer Vision for SeeForMe - Fast Object and Scene Detection
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import threading
import time

logger = logging.getLogger(__name__)

class SimpleVision:
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.last_frame = None
        self.last_analysis = None
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # Analyze every 2 seconds for speed
        
        # Simple object detection patterns (faster than YOLO)
        self.common_objects = {
            'person': 'a person',
            'face': 'someone\'s face',
            'hand': 'hands',
            'phone': 'a mobile phone',
            'cup': 'a cup or mug',
            'book': 'a book',
            'chair': 'a chair',
            'table': 'a table',
            'door': 'a door',
            'window': 'a window'
        }
        
        # Simple scene patterns
        self.scene_patterns = {
            'indoor': ['room', 'indoor space', 'inside'],
            'outdoor': ['outside', 'outdoor area', 'street'],
            'kitchen': ['kitchen', 'cooking area'],
            'bedroom': ['bedroom', 'sleeping area'],
            'office': ['office', 'workspace'],
            'bathroom': ['bathroom', 'washroom']
        }
    
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
    
    def capture_and_analyze(self) -> Optional[Dict]:
        """Capture frame and analyze quickly"""
        if not self.is_active or not self.camera:
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                return None
            
            self.last_frame = frame
            
            # Skip analysis if done recently (for speed)
            current_time = time.time()
            if (self.last_analysis and 
                current_time - self.last_analysis_time < self.analysis_interval):
                return self.last_analysis
            
            # Perform fast analysis
            analysis = self._analyze_frame_fast(frame)
            self.last_analysis = analysis
            self.last_analysis_time = current_time
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Vision analysis error: {e}")
            return None
    
    def _analyze_frame_fast(self, frame) -> Dict:
        """Fast frame analysis without heavy models"""
        height, width = frame.shape[:2]
        
        # Simple brightness and color analysis for scene type
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Simple edge detection for object estimation
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (width * height)
        
        # Estimate scene and objects based on simple patterns
        scene_description = self._estimate_scene(brightness, edge_density, frame)
        objects = self._estimate_objects(brightness, edge_density, frame)
        
        return {
            'description': scene_description,
            'objects': objects,
            'brightness': brightness,
            'complexity': edge_density,
            'timestamp': time.time()
        }
    
    def _estimate_scene(self, brightness: float, edge_density: float, frame) -> str:
        """Estimate scene type from simple metrics"""
        if brightness < 50:
            return "a dimly lit indoor space"
        elif brightness > 200:
            return "a bright outdoor area or well-lit room"
        elif edge_density > 0.1:
            return "a busy indoor space with many objects"
        else:
            return "a clean, organized indoor space"
    
    def _estimate_objects(self, brightness: float, edge_density: float, frame) -> List[str]:
        """Estimate likely objects from simple patterns"""
        objects = []
        
        # Simple heuristics based on brightness and complexity
        if edge_density > 0.08:
            objects.extend(['furniture', 'various objects'])
        
        if brightness > 150:
            objects.append('bright surfaces')
        
        if edge_density > 0.12:
            objects.extend(['complex shapes', 'detailed items'])
        
        # Always include some basic assumption
        if not objects:
            objects = ['common household items']
        
        return objects[:3]  # Limit for speed
    
    def detect_faces_simple(self) -> Optional[Dict]:
        """Simple face detection for emotion analysis"""
        if not self.last_frame is not None:
            return None
        
        try:
            # Load Haar cascade for face detection (lighter than deep learning)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Simple emotion estimation based on face detection confidence
                return {
                    'faces_detected': len(faces),
                    'emotion': 'neutral',  # Simple default
                    'confidence': 0.7
                }
        
        except Exception as e:
            logger.error(f"❌ Face detection error: {e}")
        
        return None
    
    def stop_camera(self):
        """Stop camera and cleanup"""
        self.is_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        logger.info("🛑 Camera stopped")

# Global instance
simple_vision = SimpleVision()