"""
Fast Gemma Integration for SeeForMe - Optimized for Speed
"""
import requests
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class FastGemmaConnect:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "gemma:3b"
        self.is_connected = False
        self.session = requests.Session()
        self.session.timeout = 5  # Fast timeout for speed
        
        # Test connection
        self.test_connection()
    
    def test_connection(self) -> bool:
        """Quick connection test"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if any('gemma:3b' in str(model) for model in models):
                    self.is_connected = True
                    logger.info("✅ Fast Gemma connected with gemma:3b")
                    return True
                else:
                    logger.warning("⚠️ gemma:3b model not found")
        except Exception as e:
            logger.warning(f"⚠️ Gemma not available: {e}")
            self.is_connected = False
        return False
    
    def get_fast_response(self, text: str, emotion: str = "neutral", scene: str = "", objects: list = None) -> str:
        """Generate fast emotional response"""
        if not self.is_connected:
            return self._fast_fallback(text, emotion, scene, objects)
        
        try:
            # Create optimized prompt for speed
            prompt = self._create_fast_prompt(text, emotion, scene, objects or [])
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100,  # Limit length for speed
                    "top_k": 20,
                    "top_p": 0.9
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"❌ Fast Gemma error: {e}")
        
        return self._fast_fallback(text, emotion, scene, objects)
    
    def _create_fast_prompt(self, text: str, emotion: str, scene: str, objects: list) -> str:
        """Create optimized prompt for fast responses"""
        context_parts = []
        
        if emotion != "neutral":
            context_parts.append(f"User emotion: {emotion}")
        
        if scene:
            context_parts.append(f"Environment: {scene}")
        
        if objects:
            context_parts.append(f"Objects nearby: {', '.join(objects[:3])}")  # Limit for speed
        
        context = " | ".join(context_parts) if context_parts else "Normal conversation"
        
        return f"""You are SeeForMe, an emotionally intelligent AI companion for blind users.

Context: {context}
User says: "{text}"

Respond as a caring friend with empathy and support. Keep it conversational and under 2 sentences for speed."""
    
    def _fast_fallback(self, text: str, emotion: str, scene: str, objects: list) -> str:
        """Fast fallback responses when Gemma unavailable"""
        text_lower = text.lower()
        
        # Emotional responses
        if any(word in text_lower for word in ['sad', 'bad', 'upset', 'tired', 'lonely']):
            if emotion in ['sad', 'angry', 'upset']:
                return f"I can sense you're feeling {emotion}. I'm here to listen and support you through this."
            return "I hear that you're going through a tough time. I'm here to provide comfort and care."
        
        if any(word in text_lower for word in ['happy', 'good', 'great', 'wonderful']):
            return "I'm so glad to hear you're feeling positive! That brings me joy too."
        
        # Scene-based responses
        if scene and objects:
            return f"I can see you're in {scene} with {', '.join(objects[:2])} nearby. How can I help you in this space?"
        elif scene:
            return f"I notice you're in {scene}. I'm here to help you navigate and feel comfortable."
        
        # Default caring response
        return "I'm listening carefully and I'm here to support you. What would you like to talk about?"

# Global instance for fast access
fast_gemma = FastGemmaConnect()