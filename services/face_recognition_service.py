"""
Face Recognition Service for SeeForMe
Handles face detection, encoding, and recognition using OpenCV
"""
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from services.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """Handles face detection and recognition for person identification"""
    
    def __init__(self):
        # Load face detection cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.is_initialized = True
            logger.info("✅ Face recognition service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize face recognition: {e}")
            self.face_cascade = None
            self.face_recognizer = None
            self.is_initialized = False
    
    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame and return bounding boxes"""
        if not self.is_initialized:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces.tolist()
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return []
    
    def extract_face_encoding(self, frame, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face encoding from detected face for recognition"""
        try:
            x, y, w, h = face_box
            
            # Extract face region
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for consistency
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            # Create a simple encoding using histogram
            # This is a simplified approach - in production, use proper face recognition library
            hist = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])
            
            # Normalize and flatten the histogram as encoding
            encoding = hist.flatten()
            encoding = encoding / np.linalg.norm(encoding)  # Normalize
            
            return encoding
            
        except Exception as e:
            logger.error(f"❌ Face encoding extraction failed: {e}")
            return None
    
    def process_faces_in_scene(self, frame, user_input: str = "") -> Tuple[int, List[str], str]:
        """
        Process all faces in the current frame
        Returns: (face_count, recognized_names, description)
        """
        # Handle None frame for simulation mode
        if frame is None:
            # Simulate face detection for demo purposes
            potential_new_name = memory_manager.process_name_learning(user_input, 1)
            if potential_new_name:
                return 1, [potential_new_name], f"I can see {potential_new_name} looking at you, coming toward you!"
            else:
                return 1, [], "There is one person in the view with a smiling face."
        
        faces = self.detect_faces(frame)
        face_count = len(faces)
        recognized_names = []
        unknown_faces = 0
        
        if face_count == 0:
            return 0, [], "I don't see any people in the current view."
        
        # Check if someone is introducing themselves
        potential_new_name = memory_manager.process_name_learning(user_input, face_count)
        
        for i, face_box in enumerate(faces):
            face_encoding = self.extract_face_encoding(frame, face_box)
            
            if face_encoding is not None:
                # Try to recognize the face
                recognized_name = memory_manager.recognize_face(face_encoding)
                
                if recognized_name:
                    recognized_names.append(recognized_name)
                else:
                    # Check if someone just said their name
                    if potential_new_name and unknown_faces == 0:
                        # Learn this face with the provided name
                        memory_manager.save_face_encoding(potential_new_name, face_encoding)
                        recognized_names.append(potential_new_name)
                        logger.info(f"👤 Learned new face: {potential_new_name}")
                    else:
                        unknown_faces += 1
        
        # Generate description
        description = self.generate_face_description(face_count, recognized_names, unknown_faces)
        
        return face_count, recognized_names, description
    
    def generate_face_description(self, total_faces: int, recognized_names: List[str], unknown_count: int) -> str:
        """Generate natural description of people in the scene"""
        if total_faces == 0:
            return "I don't see any people in the current view."
        
        descriptions = []
        
        # Describe recognized people
        for name in recognized_names:
            greeting = memory_manager.generate_person_greeting(name)
            descriptions.append(greeting)
        
        # Describe unknown people
        if unknown_count > 0:
            if unknown_count == 1:
                descriptions.append("There is one person I don't recognize in the view.")
            else:
                descriptions.append(f"There are {unknown_count} people I don't recognize in the view.")
        
        if not descriptions:
            if total_faces == 1:
                return "There is one person in the view."
            else:
                return f"There are {total_faces} people in the view."
        
        return " ".join(descriptions)
    
    def analyze_person_emotion_in_scene(self, frame, face_box: Tuple[int, int, int, int]) -> str:
        """Analyze emotion of a specific person in the scene"""
        try:
            x, y, w, h = face_box
            
            # Extract face region for emotion analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            
            # Simple emotion detection based on facial features
            face_area = w * h
            
            # Analyze facial proportions
            if h > 0 and w > 0:
                # Eye region (upper third)
                eye_region = face_roi[0:h//3, :]
                eye_brightness = np.mean(eye_region) if eye_region.size > 0 else 128
                
                # Mouth region (lower third)
                mouth_region = face_roi[2*h//3:h, :]
                mouth_brightness = np.mean(mouth_region) if mouth_region.size > 0 else 128
                
                # Basic emotion classification
                if mouth_brightness > eye_brightness + 15:
                    return "smiling"
                elif eye_brightness < 80:
                    return "looking sad"
                elif mouth_brightness < eye_brightness - 10:
                    return "looking concerned"
                else:
                    return "looking neutral"
            
            return "looking at you"
            
        except Exception as e:
            logger.error(f"❌ Person emotion analysis failed: {e}")
            return "looking at you"

# Global face recognition service
face_recognition_service = FaceRecognitionService()