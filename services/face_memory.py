"""
Offline Face Recognition Memory for SeeForMe
Stores face encodings and names in local JSON files
Uses face_recognition library for accurate face matching
"""
import json
import numpy as np
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path

# Try to import face_recognition, fallback to basic OpenCV if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    import cv2

logger = logging.getLogger(__name__)

class OfflineFaceMemory:
    """Manages face recognition and person memory using local JSON storage"""
    
    def __init__(self, data_dir: str = "memory_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.faces_file = self.data_dir / "known_faces.json"
        self.face_interactions_file = self.data_dir / "face_interactions.json"
        
        # Load face data
        self.known_faces = self._load_known_faces()
        self.face_interactions = self._load_face_interactions()
        
        # Initialize face detection
        self.face_cascade = None
        if not FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except Exception as e:
                logger.error(f"❌ Failed to load face cascade: {e}")
        
        logger.info(f"👤 Face memory initialized with {len(self.known_faces)} known faces")
        if FACE_RECOGNITION_AVAILABLE:
            logger.info("✅ Using face_recognition library for accurate matching")
        else:
            logger.info("⚠️ Using OpenCV fallback for basic face detection")
    
    def _load_known_faces(self) -> Dict[str, Dict]:
        """Load known faces from JSON file"""
        try:
            if self.faces_file.exists():
                with open(self.faces_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert face encodings back to numpy arrays
                    for person_name, person_data in data.items():
                        if 'face_encoding' in person_data:
                            person_data['face_encoding'] = np.array(person_data['face_encoding'])
                    return data
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load known faces: {e}")
            return {}
    
    def _save_known_faces(self):
        """Save known faces to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for person_name, person_data in self.known_faces.items():
                serializable_data[person_name] = person_data.copy()
                if 'face_encoding' in serializable_data[person_name]:
                    serializable_data[person_name]['face_encoding'] = person_data['face_encoding'].tolist()
            
            with open(self.faces_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save known faces: {e}")
    
    def _load_face_interactions(self) -> List[Dict]:
        """Load face interaction history from JSON file"""
        try:
            if self.face_interactions_file.exists():
                with open(self.face_interactions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"❌ Failed to load face interactions: {e}")
            return []
    
    def _save_face_interactions(self):
        """Save face interaction history to JSON file"""
        try:
            with open(self.face_interactions_file, 'w', encoding='utf-8') as f:
                json.dump(self.face_interactions, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save face interactions: {e}")
    
    def detect_faces_in_image(self, image) -> List[Tuple[Any, Any]]:
        """Detect faces in image and return face locations and encodings"""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                # Use face_recognition library for accurate detection
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                return list(zip(face_locations, face_encodings))
            else:
                # Fallback to OpenCV
                if self.face_cascade is None:
                    return []
                
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Create simple encodings for each face
                face_data = []
                for (x, y, w, h) in faces:
                    face_location = (y, x+w, y+h, x)  # Convert to face_recognition format
                    # Create a simple encoding using face region histogram
                    face_roi = gray[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        face_roi_resized = cv2.resize(face_roi, (64, 64))
                        hist = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])
                        face_encoding = hist.flatten()
                        face_encoding = face_encoding / np.linalg.norm(face_encoding)
                        face_data.append((face_location, face_encoding))
                
                return face_data
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return []
    
    def learn_new_face(self, image, person_name: str, relationship: str = "friend") -> bool:
        """Learn a new face and associate it with a name"""
        try:
            face_data = self.detect_faces_in_image(image)
            
            if not face_data:
                logger.warning(f"⚠️ No faces detected when learning {person_name}")
                return False
            
            # Use the first (largest) face found
            face_location, face_encoding = face_data[0]
            
            # Store the face data
            self.known_faces[person_name] = {
                'face_encoding': face_encoding,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'relationship': relationship,
                'interaction_count': 1,
                'face_location_sample': face_location
            }
            
            self._save_known_faces()
            logger.info(f"👤 Learned new face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to learn face for {person_name}: {e}")
            return False
    
    def recognize_faces_in_image(self, image, tolerance: float = 0.6) -> List[Tuple[str, Any, float]]:
        """Recognize known faces in an image"""
        try:
            face_data = self.detect_faces_in_image(image)
            recognized_faces = []
            
            for face_location, face_encoding in face_data:
                best_match_name = None
                best_match_distance = float('inf')
                
                # Compare with all known faces
                for person_name, person_data in self.known_faces.items():
                    known_encoding = person_data['face_encoding']
                    
                    if FACE_RECOGNITION_AVAILABLE:
                        # Use face_recognition's distance calculation
                        distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    else:
                        # Use simple Euclidean distance
                        distance = np.linalg.norm(face_encoding - known_encoding)
                    
                    if distance < tolerance and distance < best_match_distance:
                        best_match_distance = distance
                        best_match_name = person_name
                
                if best_match_name:
                    recognized_faces.append((best_match_name, face_location, best_match_distance))
                    
                    # Update last seen time
                    self.known_faces[best_match_name]['last_seen'] = datetime.now().isoformat()
                    self.known_faces[best_match_name]['interaction_count'] += 1
            
            if recognized_faces:
                self._save_known_faces()
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"❌ Face recognition failed: {e}")
            return []
    
    def process_scene_with_faces(self, image, user_input: str = "") -> Tuple[int, List[str], str]:
        """Process a scene, detect faces, and generate description"""
        try:
            # Check if someone is introducing themselves
            potential_new_name = self._extract_name_from_input(user_input)
            
            # Detect and recognize faces
            face_data = self.detect_faces_in_image(image)
            recognized_faces = self.recognize_faces_in_image(image)
            
            total_faces = len(face_data)
            recognized_names = [name for name, _, _ in recognized_faces]
            unknown_faces = total_faces - len(recognized_faces)
            
            # If someone said their name and there are unrecognized faces, learn the face
            if potential_new_name and unknown_faces > 0 and potential_new_name not in recognized_names:
                if self.learn_new_face(image, potential_new_name):
                    recognized_names.append(potential_new_name)
                    unknown_faces -= 1
            
            # Generate description
            description = self._generate_scene_description(total_faces, recognized_names, unknown_faces)
            
            # Log interaction
            self._log_face_interaction(recognized_names, total_faces)
            
            return total_faces, recognized_names, description
            
        except Exception as e:
            logger.error(f"❌ Scene processing failed: {e}")
            return 0, [], "I'm having trouble analyzing the scene right now."
    
    def _extract_name_from_input(self, user_input: str) -> Optional[str]:
        """Extract name from user input when they introduce themselves"""
        try:
            user_input_lower = user_input.lower()
            
            name_patterns = [
                "my name is",
                "i am",
                "i'm",
                "call me",
                "this is"
            ]
            
            for pattern in name_patterns:
                if pattern in user_input_lower:
                    pattern_index = user_input_lower.find(pattern)
                    name_part = user_input[pattern_index + len(pattern):].strip()
                    
                    words = name_part.split()
                    if words:
                        name = words[0].strip('.,!?').capitalize()
                        if len(name) > 1 and name.isalpha():
                            return name
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Name extraction failed: {e}")
            return None
    
    def _generate_scene_description(self, total_faces: int, recognized_names: List[str], unknown_count: int) -> str:
        """Generate natural description of people in the scene"""
        if total_faces == 0:
            return "I don't see any people in the current view."
        
        descriptions = []
        
        # Describe recognized people with personalized greetings
        for name in recognized_names:
            greeting = self.generate_person_greeting(name)
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
    
    def generate_person_greeting(self, person_name: str) -> str:
        """Generate appropriate greeting for recognized person"""
        try:
            if person_name not in self.known_faces:
                return f"I see {person_name} in the view."
            
            person_data = self.known_faces[person_name]
            interaction_count = person_data.get('interaction_count', 1)
            last_seen_str = person_data.get('last_seen', '')
            
            # Calculate time since last seen
            if last_seen_str:
                try:
                    last_seen = datetime.fromisoformat(last_seen_str.replace('Z', '+00:00'))
                    time_since = datetime.now() - last_seen
                    
                    if time_since.days == 0:
                        return f"Oh, there is {person_name} looking at you, coming toward you!"
                    elif time_since.days == 1:
                        return f"Hello {person_name}! I see you again. You were here yesterday."
                    elif time_since.days < 7:
                        return f"Hi {person_name}! Good to see you again after {time_since.days} days."
                    else:
                        return f"Oh, it's {person_name}! I haven't seen you in a while. Welcome back!"
                except:
                    pass
            
            return f"There is {person_name} looking at you, coming toward you!"
            
        except Exception as e:
            logger.error(f"❌ Failed to generate greeting for {person_name}: {e}")
            return f"I see {person_name} approaching you."
    
    def _log_face_interaction(self, recognized_names: List[str], total_faces: int):
        """Log face interaction for analytics"""
        try:
            interaction_entry = {
                'timestamp': datetime.now().isoformat(),
                'date': date.today().isoformat(),
                'recognized_names': recognized_names,
                'total_faces': total_faces,
                'unknown_faces': total_faces - len(recognized_names)
            }
            
            self.face_interactions.append(interaction_entry)
            
            # Keep only last 500 interactions
            if len(self.face_interactions) > 500:
                self.face_interactions = self.face_interactions[-500:]
            
            self._save_face_interactions()
            
        except Exception as e:
            logger.error(f"❌ Failed to log face interaction: {e}")
    
    def get_known_people(self) -> List[str]:
        """Get list of all known people"""
        return list(self.known_faces.keys())
    
    def get_person_info(self, person_name: str) -> Dict[str, Any]:
        """Get information about a specific person"""
        return self.known_faces.get(person_name, {})
    
    def forget_person(self, person_name: str) -> bool:
        """Remove a person from memory"""
        try:
            if person_name in self.known_faces:
                del self.known_faces[person_name]
                self._save_known_faces()
                logger.info(f"🗑️ Removed {person_name} from face memory")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to forget {person_name}: {e}")
            return False
    
    def get_face_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about face memory"""
        try:
            total_interactions = len(self.face_interactions)
            recent_interactions = len([
                interaction for interaction in self.face_interactions
                if interaction['date'] >= (date.today() - timedelta(days=7)).isoformat()
            ])
            
            return {
                'total_known_people': len(self.known_faces),
                'total_interactions': total_interactions,
                'recent_interactions': recent_interactions,
                'known_people': list(self.known_faces.keys()),
                'face_recognition_library': FACE_RECOGNITION_AVAILABLE
            }
        except Exception as e:
            logger.error(f"❌ Failed to get face memory stats: {e}")
            return {}

# Global offline face memory instance
offline_face_memory = OfflineFaceMemory()