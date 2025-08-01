"""
Memory Manager for SeeForMe
Handles emotional continuity, conversation history, and face recognition
"""
import json
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from app import db
from models import UserSession, ConversationHistory, FaceRecognition, EmotionalMemory
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages all memory functions for emotional continuity and face recognition"""
    
    def __init__(self):
        self.current_user = "friend"
        self.face_encodings_cache = {}  # In-memory cache for faster lookup
        self.load_face_encodings()
    
    def save_conversation(self, user_name: str, user_input: str, ai_response: str, emotion_detected: str = None):
        """Save conversation to database for memory continuity"""
        try:
            conversation = ConversationHistory(
                user_name=user_name,
                user_input=user_input,
                ai_response=ai_response,
                emotion_detected=emotion_detected,
                timestamp=datetime.utcnow()
            )
            db.session.add(conversation)
            db.session.commit()
            logger.info(f"💾 Saved conversation for {user_name}")
        except Exception as e:
            logger.error(f"❌ Failed to save conversation: {e}")
            db.session.rollback()
    
    def save_emotional_state(self, user_name: str, emotion: str, intensity: float = 0.5, context: str = None):
        """Save user's emotional state for future reference"""
        try:
            today = date.today()
            
            # Check if emotional memory exists for today
            existing_memory = EmotionalMemory.query.filter_by(
                user_name=user_name,
                date=today
            ).first()
            
            if existing_memory:
                # Update existing emotional state
                existing_memory.dominant_emotion = emotion
                existing_memory.emotion_intensity = intensity
                existing_memory.context_description = context
                existing_memory.resolution_status = "ongoing"
            else:
                # Create new emotional memory
                emotional_memory = EmotionalMemory(
                    user_name=user_name,
                    date=today,
                    dominant_emotion=emotion,
                    emotion_intensity=intensity,
                    context_description=context,
                    resolution_status="ongoing"
                )
                db.session.add(emotional_memory)
            
            db.session.commit()
            logger.info(f"💭 Saved emotional state: {emotion} for {user_name}")
        except Exception as e:
            logger.error(f"❌ Failed to save emotional state: {e}")
            db.session.rollback()
    
    def get_emotional_continuity(self, user_name: str) -> str:
        """Get emotional context from previous sessions for continuity"""
        try:
            # Get yesterday's emotional state
            yesterday = date.today() - timedelta(days=1)
            yesterday_emotion = EmotionalMemory.query.filter_by(
                user_name=user_name,
                date=yesterday
            ).first()
            
            if yesterday_emotion and yesterday_emotion.resolution_status != "resolved":
                emotion = yesterday_emotion.dominant_emotion
                context = yesterday_emotion.context_description or ""
                
                if emotion == "sad":
                    return f"Yesterday you weren't feeling well and seemed sad. How are you feeling today? Are you alright now? Share with me what you did today."
                elif emotion == "angry":
                    return f"Yesterday you seemed upset about something. I hope things are better today. How are you feeling now?"
                elif emotion == "worried":
                    return f"Yesterday you seemed concerned about something. Is everything okay today? How did things go?"
                elif emotion == "happy":
                    return f"Yesterday you were in such a good mood! I hope your day continued to be wonderful. How are you feeling today?"
                else:
                    return f"Yesterday you were feeling {emotion}. How are you doing today? Tell me about your day."
            
            # Check for recent conversations in the last 24 hours
            recent_conversations = ConversationHistory.query.filter(
                ConversationHistory.user_name == user_name,
                ConversationHistory.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ).order_by(ConversationHistory.timestamp.desc()).limit(3).all()
            
            if recent_conversations:
                last_emotion = recent_conversations[0].emotion_detected
                if last_emotion and last_emotion in ["sad", "angry", "worried"]:
                    return f"Last time we talked, you seemed {last_emotion}. I've been thinking about you. How are you feeling now?"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get emotional continuity: {e}")
            return None
    
    def save_face_encoding(self, person_name: str, face_encoding: np.ndarray, relationship: str = "unknown"):
        """Save face encoding for person recognition"""
        try:
            # Convert numpy array to JSON string
            encoding_json = json.dumps(face_encoding.tolist())
            
            # Check if person already exists
            existing_person = FaceRecognition.query.filter_by(person_name=person_name).first()
            
            if existing_person:
                # Update existing person
                existing_person.face_encoding = encoding_json
                existing_person.last_seen = datetime.utcnow()
                existing_person.interaction_count += 1
            else:
                # Create new person entry
                face_record = FaceRecognition(
                    person_name=person_name,
                    face_encoding=encoding_json,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    interaction_count=1,
                    relationship_notes=relationship
                )
                db.session.add(face_record)
            
            db.session.commit()
            
            # Update cache
            self.face_encodings_cache[person_name] = face_encoding
            logger.info(f"👤 Saved face encoding for {person_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save face encoding: {e}")
            db.session.rollback()
    
    def load_face_encodings(self):
        """Load all face encodings into memory cache for faster recognition"""
        try:
            all_faces = FaceRecognition.query.all()
            self.face_encodings_cache = {}
            
            for face_record in all_faces:
                encoding_list = json.loads(face_record.face_encoding)
                encoding_array = np.array(encoding_list)
                self.face_encodings_cache[face_record.person_name] = encoding_array
                
            logger.info(f"👥 Loaded {len(self.face_encodings_cache)} face encodings into cache")
        except Exception as e:
            logger.error(f"❌ Failed to load face encodings: {e}")
    
    def recognize_face(self, face_encoding: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Recognize a person from their face encoding"""
        try:
            if not self.face_encodings_cache:
                return None
            
            # Calculate distances to all known faces
            for person_name, known_encoding in self.face_encodings_cache.items():
                # Calculate Euclidean distance
                distance = np.linalg.norm(face_encoding - known_encoding)
                
                if distance < tolerance:
                    # Update last seen time
                    person_record = FaceRecognition.query.filter_by(person_name=person_name).first()
                    if person_record:
                        person_record.last_seen = datetime.utcnow()
                        person_record.interaction_count += 1
                        db.session.commit()
                    
                    logger.info(f"👤 Recognized person: {person_name} (distance: {distance:.3f})")
                    return person_name
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Face recognition failed: {e}")
            return None
    
    def generate_person_greeting(self, person_name: str) -> str:
        """Generate appropriate greeting for recognized person"""
        try:
            person_record = FaceRecognition.query.filter_by(person_name=person_name).first()
            
            if person_record:
                interaction_count = person_record.interaction_count
                last_seen = person_record.last_seen
                time_since_last_seen = datetime.utcnow() - last_seen
                
                if time_since_last_seen.days == 0:
                    return f"Oh, there is {person_name} looking at you, coming toward you!"
                elif time_since_last_seen.days == 1:
                    return f"Hello {person_name}! I see you again. You were here yesterday."
                elif time_since_last_seen.days < 7:
                    return f"Hi {person_name}! Good to see you again after {time_since_last_seen.days} days."
                else:
                    return f"Oh, it's {person_name}! I haven't seen you in a while. Welcome back!"
            else:
                return f"There is {person_name} looking at you, coming toward you!"
                
        except Exception as e:
            logger.error(f"❌ Failed to generate person greeting: {e}")
            return f"I see {person_name} approaching you."
    
    def process_name_learning(self, user_input: str, detected_faces_count: int) -> Optional[str]:
        """Process when someone says their name to learn new faces"""
        try:
            user_input_lower = user_input.lower()
            
            # Check if someone is introducing themselves
            name_patterns = [
                "my name is",
                "i am",
                "i'm",
                "call me",
                "this is"
            ]
            
            for pattern in name_patterns:
                if pattern in user_input_lower:
                    # Extract the name after the pattern
                    pattern_index = user_input_lower.find(pattern)
                    name_part = user_input[pattern_index + len(pattern):].strip()
                    
                    # Extract first word as name (simple extraction)
                    words = name_part.split()
                    if words:
                        name = words[0].strip('.,!?').capitalize()
                        
                        # Validate name (basic validation)
                        if len(name) > 1 and name.isalpha():
                            return name
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Name learning failed: {e}")
            return None
    
    def get_conversation_context(self, user_name: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation history for context"""
        try:
            recent_conversations = ConversationHistory.query.filter_by(
                user_name=user_name
            ).order_by(
                ConversationHistory.timestamp.desc()
            ).limit(limit).all()
            
            context = []
            for conv in reversed(recent_conversations):  # Reverse to get chronological order
                context.append({
                    'user_input': conv.user_input,
                    'ai_response': conv.ai_response,
                    'emotion': conv.emotion_detected,
                    'timestamp': conv.timestamp
                })
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation context: {e}")
            return []

# Global memory manager instance
memory_manager = MemoryManager()