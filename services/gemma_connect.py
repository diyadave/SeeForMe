#!/usr/bin/env python3
"""
Optimized Gemma 3n Integration for Offline Accessibility App
Fast, reliable connection to Ollama with intelligent fallbacks and caching
"""

import requests
import json
import time
import threading
import queue
import logging
import sys
import platform
from typing import Optional, Dict, Any, Callable, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib

# Fix Unicode logging on Windows
if platform.system() == "Windows":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logger = logging.getLogger(__name__)

class GemmaConnector:
    """Optimized connector for Gemma 3n model via Ollama"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model_name: str = "gemma2:2b"):
        self.base_url = base_url
        self.model_name = model_name
        self.generate_endpoint = f"{base_url}/api/generate"
        
        # Connection status
        self.is_connected = False
        self.is_model_ready = False
        self.connection_attempts = 0
        self.max_connection_attempts = 10
        self.connection_retry_delay = 1.0
        self.fast_retry_delay = 0.3
        self.fast_timeout = 5.0
        self.stop_connection_attempts = False
        
        # Performance tracking
        self.average_response_time = 0.0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        
        # Threading and caching
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.response_cache = {}
        self.cache_max_size = 50
        
        # Session with keep-alive
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        })
        
        # Configuration
        self.default_timeout = 15
        self.max_retries = 2
        self.warmup_completed = False
        
        # Start connection thread
        self.connection_thread = threading.Thread(
            target=self.persistent_connection_worker,
            daemon=True
        )
        self.connection_thread.start()
    
    def persistent_connection_worker(self):
        """Persistent worker that keeps trying to connect to Gemma"""
        while (not self.stop_connection_attempts and 
               self.connection_attempts < self.max_connection_attempts):
            try:
                self.connection_attempts += 1
                
                # Test connection
                if self.test_connection():
                    logger.info("✅ Basic connection established")
                    
                    # Warm up model
                    if self.warmup_model():
                        logger.info("🔥 Gemma ready for requests")
                        self.is_connected = True
                        self.is_model_ready = True
                        return
                
                if not self.stop_connection_attempts:
                    delay = self.fast_retry_delay if self.connection_attempts < 3 else self.connection_retry_delay
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Connection attempt {self.connection_attempts} failed: {str(e)}")
                time.sleep(self.connection_retry_delay)
        
        logger.error("❌ Max connection attempts reached. Running in fallback mode.")
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self.fast_timeout
            )
            
            if response.ok:
                models = response.json().get('models', [])
                return any(self.model_name in model.get('name', '') for model in models)
            return False
            
        except Exception as e:
            logger.debug(f"Connection test failed: {str(e)}")
            return False
    
    def warmup_model(self) -> bool:
        """Warm up the model with a simple prompt"""
        try:
            warmup_prompt = "Hello, this is a warmup message. Please respond with just 'OK'."
            
            response = self.session.post(
                self.generate_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": warmup_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "max_tokens": 5}
                },
                timeout=self.fast_timeout
            )
            
            if response.ok and 'OK' in response.json().get('response', ''):
                self.warmup_completed = True
                return True
            return False
            
        except Exception as e:
            logger.debug(f"Model warmup failed: {str(e)}")
            return False
    
    def generate_cache_key(self, prompt: str, context: Dict) -> str:
        """Generate consistent cache key"""
        cache_str = f"{prompt}_{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get response from cache if available"""
        cached = self.response_cache.get(cache_key)
        if cached and (time.time() - cached['timestamp'] < 300):  # 5 minute cache
            self.cache_hits += 1
            return cached['response']
        return None
    
    def cache_response(self, cache_key: str, response: str):
        """Cache response with size limit"""
        if len(self.response_cache) >= self.cache_max_size:
            oldest_key = min(self.response_cache.keys(), key=lambda k: self.response_cache[k]['timestamp'])
            del self.response_cache[oldest_key]
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def build_emotional_prompt(self, user_text: str, context: Dict) -> str:
        """Build emotionally aware prompt for Gemma"""
        user_name = context.get('user_name', 'friend')
        current_emotion = context.get('current_emotion', 'Neutral')
        scene_info = context.get('scene_info', 'indoor space')
        language = context.get('language', 'en')
        
        # Emotion guidance
        emotion_guidance = {
            'Happy': "The user seems happy. Celebrate with them and match their positive energy!",
            'Sad': "The user seems sad. Be compassionate, supportive, and offer comfort.",
            'Angry': "The user seems frustrated or angry. Be calm, understanding, and de-escalating.",
            'Fear': "The user seems afraid or anxious. Be reassuring and calming.",
            'Surprise': "The user seems surprised. Acknowledge their surprise and be helpful.",
            'Neutral': "The user's emotion is neutral. Be warm, engaging, and helpful."
        }
        
        return f"""You are a compassionate AI assistant helping a blind user named {user_name}.
Current emotional state: {current_emotion}
Current environment: {scene_info}
Guidance: {emotion_guidance.get(current_emotion, 'Be kind and helpful.')}

User message: "{user_text}"

Respond with:
1. Emotional awareness and empathy
2. Personalized response using their name
3. 1-2 concise, clear sentences
4. Natural, conversational tone
5. If asked about surroundings, use the environment context

Response:"""
    
    def generate_response(self, user_text: str, context: Dict, timeout: int = None) -> Optional[str]:
        """Generate response with fast fallback"""
        if not self.is_model_ready:
            return self.get_fallback_response(user_text, context)
        
        timeout = timeout or self.default_timeout
        cache_key = self.generate_cache_key(user_text, context)
        
        # Check cache first
        if cached := self.get_cached_response(cache_key):
            return cached
        
        # Build prompt
        prompt = self.build_emotional_prompt(user_text, context)
        
        def generate_worker():
            """Worker function for threaded generation"""
            self.total_requests += 1
            start_time = time.time()
            
            for attempt in range(self.max_retries):
                try:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 150
                        }
                    }
                    
                    response = self.session.post(
                        self.generate_endpoint,
                        json=payload,
                        timeout=timeout
                    )
                    
                    if response.ok:
                        result = response.json()
                        if 'response' in result:
                            response_text = result['response'].strip()
                            if response_text:
                                # Update metrics and cache
                                response_time = time.time() - start_time
                                self.update_metrics(response_time, True)
                                self.cache_response(cache_key, response_text)
                                return response_text
                    
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
            
            # All retries failed
            self.update_metrics(0, False)
            return self.get_fallback_response(user_text, context)
        
        try:
            future = self.executor.submit(generate_worker)
            return future.result(timeout=timeout + 2)
        except FutureTimeoutError:
            logger.warning("Generation timeout")
            return self.get_fallback_response(user_text, context)
    
    def get_fallback_response(self, user_text: str, context: Dict) -> str:
        """Get appropriate fallback response"""
        text_lower = user_text.lower()
        user_name = context.get('user_name', 'there')
        current_emotion = context.get('current_emotion', 'Neutral')
        
        # Emotion-aware fallback responses
        if current_emotion == 'Sad':
            return f"{user_name}, I'm here for you. Even though I'm having trouble with my advanced responses right now, I want you to know I'm listening and I care."
        elif current_emotion == 'Happy':
            return f"That's wonderful to hear, {user_name}! I wish I could share in your joy with a better response, but I'm having some technical difficulties."
        elif current_emotion == 'Angry':
            return f"I understand you might be frustrated, {user_name}. I'm experiencing some technical issues but I'm here to help as best I can."
        
        # Simple keyword matching for fallback
        if 'name' in text_lower:
            return f"Hello {user_name}! Nice to meet you."
        elif any(e in text_lower for e in ['sad', 'upset', 'unhappy']):
            return f"{user_name}, I'm sorry you're feeling this way. I'm here to help."
        elif any(e in text_lower for e in ['happy', 'joy', 'excited']):
            return f"That's wonderful {user_name}! I'm glad you're feeling good."
        elif any(q in text_lower for q in ['what', 'describe', 'see', 'scene']):
            return "Let me try to process what I'm seeing for you..."
        elif any(q in text_lower for q in ['hello', 'hi', 'hey']):
            return f"Hello {user_name}! How can I assist you today?"
        
        return f"I understand, {user_name}. How can I assist you further?"
    
    def update_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        if success:
            self.successful_requests += 1
            # Exponential moving average for response time
            if self.average_response_time == 0:
                self.average_response_time = response_time
            else:
                self.average_response_time = 0.2 * response_time + 0.8 * self.average_response_time
        else:
            self.failed_requests += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'connected': self.is_connected,
            'model_ready': self.is_model_ready,
            'connection_attempts': self.connection_attempts,
            'total_requests': self.total_requests,
            'success_rate': f"{success_rate:.1f}%",
            'avg_response_time': f"{self.average_response_time:.2f}s",
            'cache_size': len(self.response_cache),
            'cache_hits': self.cache_hits
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_connection_attempts = True
        self.executor.shutdown(wait=False)
        self.session.close()
        logger.info("Gemma connector cleaned up")

# Test function
if __name__ == "__main__":
    connector = GemmaConnector()
    
    # Wait for connection
    time.sleep(3)
    
    # Test conversation
    context = {
        'user_name': 'Alex',
        'current_emotion': 'Happy',
        'scene_info': 'living room with couch and TV',
        'language': 'en'
    }
    
    response = connector.generate_response("Hello, how are you today?", context)
    print(f"Response: {response}")
    
    # Cleanup
    connector.cleanup()
