#!/usr/bin/env python3
"""
Scene Detector - Comprehensive scene analysis using YOLOv8n and Places365
Combines object detection, scene classification, and people detection
"""

import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import os

logger = logging.getLogger(__name__)

class SceneDetector:
    """Comprehensive scene analysis using multiple AI models"""
    
    def __init__(self):
        self.is_initialized = False
        
        # Model components
        self.yolo_model = None
        self.places_model = None
        
        # COCO classes for YOLOv8
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Places365 categories (loaded from file)
        self.places_categories = self.load_places_categories()
        
        # Initialize models
        self.initialize_yolo()
        self.initialize_places()
        
        logger.info("🔍 Scene detector initialized")
    
    def load_places_categories(self) -> List[str]:
        """Load Places365 categories from file"""
        try:
            categories_path = 'attached_assets/categories_places365_1753972140816.txt'
            if os.path.exists(categories_path):
                with open(categories_path, 'r') as f:
                    categories = []
                    for line in f:
                        parts = line.strip().split(' ')
                        if len(parts) >= 2:
                            category = parts[0].replace('/', ' ').replace('_', ' ').strip()
                            categories.append(category)
                    return categories
            else:
                logger.warning("⚠️ Places365 categories file not found")
                return self.get_default_places_categories()
        except Exception as e:
            logger.error(f"❌ Failed to load Places365 categories: {e}")
            return self.get_default_places_categories()
    
    def get_default_places_categories(self) -> List[str]:
        """Get default Places365 categories"""
        return [
            'bedroom', 'living room', 'kitchen', 'bathroom', 'dining room',
            'office', 'classroom', 'library', 'restaurant', 'cafe',
            'park', 'street', 'beach', 'forest', 'mountain',
            'garden', 'yard', 'plaza', 'building', 'house'
        ]
    
    def initialize_yolo(self):
        """Initialize YOLOv8n object detection model"""
        try:
            from ultralytics import YOLO
            
            # Look for YOLOv8n model
            model_paths = [
                'attached_assets/yolov8n_1753972140820.pt',
                'models/yolov8n.pt',
                'yolov8n.pt'
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                self.yolo_model = YOLO(model_path)
                logger.info(f"✅ YOLOv8n model loaded: {model_path}")
            else:
                # Try to download YOLOv8n if not found
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("✅ YOLOv8n model downloaded and loaded")
                
            return True
            
        except ImportError:
            logger.warning("⚠️ Ultralytics not available, using fallback object detection")
            return False
        except Exception as e:
            logger.error(f"❌ YOLOv8n initialization failed: {e}")
            return False
    
    def initialize_places(self):
        """Initialize Places365 scene classification model"""
        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision import models
            
            # Look for Places365 model
            model_paths = [
                'attached_assets/resnet18_places365.pth_1753972140818.tar',
                'models/resnet18_places365.pth',
                'resnet18_places365.pth'
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                # Load Places365 model
                self.places_model = models.resnet18(num_classes=365)
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                self.places_model.load_state_dict(state_dict)
                self.places_model.eval()
                
                # Define preprocessing
                self.places_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                logger.info(f"✅ Places365 model loaded: {model_path}")
                return True
            else:
                logger.warning("⚠️ Places365 model not found, using fallback scene detection")
                return False
                
        except ImportError:
            logger.warning("⚠️ PyTorch not available, using fallback scene detection")
            return False
        except Exception as e:
            logger.error(f"❌ Places365 initialization failed: {e}")
            return False
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using YOLOv8n"""
        if self.yolo_model is None:
            return self.fallback_object_detection(image)
        
        try:
            # Run YOLO inference
            results = self.yolo_model(image, verbose=False)
            
            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get box coordinates
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        
                        # Get class and confidence
                        class_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        if confidence > 0.3:  # Confidence threshold
                            objects.append({
                                'class': self.coco_classes[class_id],
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
            
            return objects
            
        except Exception as e:
            logger.error(f"❌ Object detection failed: {e}")
            return self.fallback_object_detection(image)
    
    def fallback_object_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback object detection using simple image analysis"""
        # Simple fallback that detects basic shapes and colors
        objects = []
        
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect common objects based on color and shape
            # This is a very basic fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangular objects (tables, books, etc.)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours[:5]:  # Limit to 5 largest contours
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Classify based on aspect ratio
                    if 0.8 < aspect_ratio < 1.2:
                        obj_class = 'object'
                    elif aspect_ratio > 1.5:
                        obj_class = 'table'
                    else:
                        obj_class = 'item'
                    
                    objects.append({
                        'class': obj_class,
                        'confidence': 0.5,
                        'bbox': [x, y, x+w, y+h],
                        'fallback': True
                    })
            
        except Exception as e:
            logger.error(f"❌ Fallback object detection failed: {e}")
        
        return objects
    
    def classify_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify scene using Places365"""
        if self.places_model is None:
            return self.fallback_scene_classification(image)
        
        try:
            import torch
            
            # Preprocess image
            input_tensor = self.places_transform(image)
            input_batch = input_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.places_model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            scene_predictions = []
            for i in range(5):
                category_idx = int(top5_catid[i])
                confidence = float(top5_prob[i])
                
                if category_idx < len(self.places_categories):
                    scene_predictions.append({
                        'scene': self.places_categories[category_idx],
                        'confidence': confidence
                    })
            
            # Return top prediction
            if scene_predictions:
                return {
                    'scene_type': scene_predictions[0]['scene'],
                    'confidence': scene_predictions[0]['confidence'],
                    'all_predictions': scene_predictions
                }
            else:
                return self.fallback_scene_classification(image)
                
        except Exception as e:
            logger.error(f"❌ Scene classification failed: {e}")
            return self.fallback_scene_classification(image)
    
    def fallback_scene_classification(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback scene classification using basic image analysis"""
        try:
            # Analyze image properties for basic scene classification
            height, width = image.shape[:2]
            
            # Calculate brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Calculate color distribution
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Simple heuristics for scene classification
            if brightness > 200:
                scene_type = 'bright indoor space'
            elif brightness < 50:
                scene_type = 'dark environment'
            elif np.mean(hsv[:, :, 1]) > 100:  # High saturation
                scene_type = 'outdoor area'
            else:
                scene_type = 'indoor space'
            
            return {
                'scene_type': scene_type,
                'confidence': 0.6,
                'brightness': float(brightness),
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback scene classification failed: {e}")
            return {
                'scene_type': 'unknown environment',
                'confidence': 0.3,
                'fallback': True
            }
    
    def detect_people(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract people from detected objects"""
        people = []
        
        for obj in objects:
            if obj['class'] == 'person':
                people.append({
                    'bbox': obj['bbox'],
                    'confidence': obj['confidence']
                })
        
        return people
    
    def analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive scene analysis"""
        if image is None:
            return {}
        
        try:
            # Detect objects
            objects = self.detect_objects(image)
            
            # Classify scene
            scene_info = self.classify_scene(image)
            
            # Extract people
            people = self.detect_people(objects)
            
            # Analyze lighting and time of day
            lighting_info = self.analyze_lighting(image)
            
            # Create object summary
            object_classes = [obj['class'] for obj in objects if obj['confidence'] > 0.4]
            unique_objects = list(set(object_classes))
            
            return {
                'scene_type': scene_info.get('scene_type', 'unknown'),
                'scene_confidence': scene_info.get('confidence', 0.0),
                'objects': unique_objects,
                'objects_detailed': objects,
                'people': people,
                'people_count': len(people),
                'lighting': lighting_info,
                'total_objects': len(objects)
            }
            
        except Exception as e:
            logger.error(f"❌ Scene analysis failed: {e}")
            return {
                'scene_type': 'unknown',
                'objects': [],
                'people': [],
                'people_count': 0,
                'error': str(e)
            }
    
    def analyze_lighting(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting conditions and estimate time of day"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate color temperature (simplified)
            b, g, r = cv2.split(image)
            color_temp_ratio = np.mean(r) / (np.mean(b) + 1)  # Avoid division by zero
            
            # Estimate time of day
            if brightness > 180:
                time_estimate = 'bright daylight'
            elif brightness > 120:
                if color_temp_ratio > 1.2:
                    time_estimate = 'golden hour / sunset'
                else:
                    time_estimate = 'daytime'
            elif brightness > 60:
                time_estimate = 'evening / dusk'
            else:
                time_estimate = 'night / artificial lighting'
            
            # Lighting quality
            if brightness > 150:
                quality = 'excellent'
            elif brightness > 100:
                quality = 'good'
            elif brightness > 50:
                quality = 'moderate'
            else:
                quality = 'poor'
            
            return {
                'brightness': float(brightness),
                'quality': quality,
                'time_estimate': time_estimate,
                'color_temperature_ratio': float(color_temp_ratio)
            }
            
        except Exception as e:
            logger.error(f"❌ Lighting analysis failed: {e}")
            return {
                'brightness': 0.0,
                'quality': 'unknown',
                'time_estimate': 'unknown'
            }
    
    def generate_scene_description(self, analysis: Dict[str, Any]) -> str:
        """Generate natural language scene description"""
        try:
            scene_type = analysis.get('scene_type', 'an area')
            objects = analysis.get('objects', [])
            people_count = analysis.get('people_count', 0)
            lighting = analysis.get('lighting', {})
            
            description_parts = []
            
            # Scene type
            description_parts.append(f"You are in {scene_type}.")
            
            # Lighting
            time_estimate = lighting.get('time_estimate', '')
            if time_estimate:
                description_parts.append(f"The lighting suggests it's {time_estimate}.")
            
            # People
            if people_count == 1:
                description_parts.append("There is one person visible.")
            elif people_count > 1:
                description_parts.append(f"There are {people_count} people visible.")
            
            # Objects
            if objects:
                if len(objects) <= 3:
                    obj_list = ', '.join(objects)
                    description_parts.append(f"I can see: {obj_list}.")
                else:
                    description_parts.append(f"I can see several items including {objects[0]}, {objects[1]}, and {len(objects)-2} other objects.")
            
            return ' '.join(description_parts)
            
        except Exception as e:
            logger.error(f"❌ Scene description generation failed: {e}")
            return "I can see your surroundings, but I'm having difficulty describing the details right now."
    
    def get_status(self) -> Dict[str, Any]:
        """Get current detector status"""
        return {
            'status': 'ready' if self.is_initialized else 'fallback',
            'yolo_model': self.yolo_model is not None,
            'places_model': self.places_model is not None,
            'coco_classes_count': len(self.coco_classes),
            'places_categories_count': len(self.places_categories)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up scene detector...")
        
        self.yolo_model = None
        self.places_model = None
        self.is_initialized = False
        
        logger.info("✅ Scene detector cleanup completed")