"""
Offline Memory Manager for SeeForMe
Stores emotional states and conversation history in local JSON files
No database dependencies - fully offline and portable
"""
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class OfflineMemoryManager:
    """Manages emotional continuity and conversation history using local JSON files"""
    
    def __init__(self, data_dir: str = "memory_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths for different data types
        self.conversations_file = self.data_dir / "conversations.json"
        self.emotions_file = self.data_dir / "emotional_states.json"
        self.user_profiles_file = self.data_dir / "user_profiles.json"
        
        # In-memory cache for faster access
        self.conversations_cache = self._load_conversations()
        self.emotions_cache = self._load_emotions()
        self.user_profiles_cache = self._load_user_profiles()
        
        logger.info(f"💾 Offline memory manager initialized at {self.data_dir}")
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversation history from JSON file"""
        try:
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"❌ Failed to load conversations: {e}")
            return []
    
    def _save_conversations(self):
        """Save conversation history to JSON file"""
        try:
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations_cache, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save conversations: {e}")
    
    def _load_emotions(self) -> Dict[str, Dict]:
        """Load emotional states from JSON file"""
        try:
            if self.emotions_file.exists():
                with open(self.emotions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load emotions: {e}")
            return {}
    
    def _save_emotions(self):
        """Save emotional states to JSON file"""
        try:
            with open(self.emotions_file, 'w', encoding='utf-8') as f:
                json.dump(self.emotions_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Failed to save emotions: {e}")
    
    def _load_user_profiles(self) -> Dict[str, Dict]:
        """Load user profiles from JSON file"""
        try:
            if self.user_profiles_file.exists():
                with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load user profiles: {e}")
            return {}
    
    def _save_user_profiles(self):
        """Save user profiles to JSON file"""
        try:
            with open(self.user_profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles_cache, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save user profiles: {e}")
    
    def save_conversation(self, user_name: str, user_input: str, ai_response: str, emotion_detected: str = None):
        """Save conversation to local JSON for memory continuity"""
        try:
            conversation_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_name': user_name,
                'user_input': user_input,
                'ai_response': ai_response,
                'emotion_detected': emotion_detected,
                'date': date.today().isoformat()
            }
            
            self.conversations_cache.append(conversation_entry)
            
            # Keep only last 1000 conversations to prevent file growth
            if len(self.conversations_cache) > 1000:
                self.conversations_cache = self.conversations_cache[-1000:]
            
            self._save_conversations()
            logger.info(f"💾 Saved conversation for {user_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save conversation: {e}")
    
    def save_emotional_state(self, user_name: str, emotion: str, intensity: float = 0.5, context: str = None):
        """Save user's emotional state for future reference"""
        try:
            today = date.today().isoformat()
            
            if user_name not in self.emotions_cache:
                self.emotions_cache[user_name] = {}
            
            # Update or create emotional state for today
            self.emotions_cache[user_name][today] = {
                'emotion': emotion,
                'intensity': intensity,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'resolution_status': 'ongoing'
            }
            
            self._save_emotions()
            logger.info(f"💭 Saved emotional state: {emotion} for {user_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save emotional state: {e}")
    
    def get_emotional_continuity(self, user_name: str) -> Optional[str]:
        """Get emotional context from previous sessions for continuity"""
        try:
            if user_name not in self.emotions_cache:
                return None
            
            user_emotions = self.emotions_cache[user_name]
            
            # Check yesterday's emotional state
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            if yesterday in user_emotions:
                yesterday_emotion = user_emotions[yesterday]
                emotion = yesterday_emotion['emotion']
                resolution_status = yesterday_emotion.get('resolution_status', 'ongoing')
                
                if resolution_status != 'resolved':
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
            
            # Check for recent emotional conversations in last 24 hours
            recent_conversations = self.get_conversation_context(user_name, limit=3)
            if recent_conversations:
                last_emotion = None
                for conv in reversed(recent_conversations):
                    if conv.get('emotion_detected') and conv['emotion_detected'] in ["sad", "angry", "worried"]:
                        last_emotion = conv['emotion_detected']
                        break
                
                if last_emotion:
                    return f"Last time we talked, you seemed {last_emotion}. I've been thinking about you. How are you feeling now?"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get emotional continuity: {e}")
            return None
    
    def get_conversation_context(self, user_name: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation history for context"""
        try:
            # Filter conversations for this user from the last 7 days
            cutoff_date = (date.today() - timedelta(days=7)).isoformat()
            
            user_conversations = [
                conv for conv in self.conversations_cache
                if conv['user_name'] == user_name and conv['date'] >= cutoff_date
            ]
            
            # Sort by timestamp and return latest conversations
            user_conversations.sort(key=lambda x: x['timestamp'], reverse=True)
            return user_conversations[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation context: {e}")
            return []
    
    def update_user_profile(self, user_name: str, profile_data: Dict[str, Any]):
        """Update user profile information"""
        try:
            if user_name not in self.user_profiles_cache:
                self.user_profiles_cache[user_name] = {
                    'created_date': date.today().isoformat(),
                    'last_seen': datetime.now().isoformat()
                }
            
            self.user_profiles_cache[user_name].update(profile_data)
            self.user_profiles_cache[user_name]['last_seen'] = datetime.now().isoformat()
            
            self._save_user_profiles()
            logger.info(f"👤 Updated profile for {user_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update user profile: {e}")
    
    def get_user_profile(self, user_name: str) -> Dict[str, Any]:
        """Get user profile information"""
        return self.user_profiles_cache.get(user_name, {})
    
    def mark_emotion_resolved(self, user_name: str, date_str: str = None):
        """Mark an emotional state as resolved"""
        try:
            if date_str is None:
                date_str = date.today().isoformat()
            
            if user_name in self.emotions_cache and date_str in self.emotions_cache[user_name]:
                self.emotions_cache[user_name][date_str]['resolution_status'] = 'resolved'
                self._save_emotions()
                logger.info(f"✅ Marked emotion resolved for {user_name} on {date_str}")
                
        except Exception as e:
            logger.error(f"❌ Failed to mark emotion resolved: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to prevent files from growing too large"""
        try:
            cutoff_date = (date.today() - timedelta(days=days_to_keep)).isoformat()
            
            # Clean old conversations
            self.conversations_cache = [
                conv for conv in self.conversations_cache
                if conv['date'] >= cutoff_date
            ]
            
            # Clean old emotions
            for user_name in list(self.emotions_cache.keys()):
                user_emotions = self.emotions_cache[user_name]
                self.emotions_cache[user_name] = {
                    date_str: emotion_data
                    for date_str, emotion_data in user_emotions.items()
                    if date_str >= cutoff_date
                }
                
                # Remove user if no recent emotions
                if not self.emotions_cache[user_name]:
                    del self.emotions_cache[user_name]
            
            self._save_conversations()
            self._save_emotions()
            
            logger.info(f"🧹 Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
    
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
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memory data"""
        try:
            return {
                'total_conversations': len(self.conversations_cache),
                'total_users': len(self.user_profiles_cache),
                'users_with_emotions': len(self.emotions_cache),
                'data_directory': str(self.data_dir),
                'last_conversation': self.conversations_cache[-1]['timestamp'] if self.conversations_cache else None
            }
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {}

# Global offline memory manager instance
offline_memory_manager = OfflineMemoryManager()