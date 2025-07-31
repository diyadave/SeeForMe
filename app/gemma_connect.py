#!/usr/bin/env python3
"""
Gemma Connector - Integration with Gemma 3n LLM via Ollama
Provides intelligent, empathetic responses for vision accessibility
"""

import logging
import requests
import json
import time
import threading
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class GemmaConnector:
    """Gemma 3n LLM integration for intelligent responses"""
    
    def __init__(self):
        self.is_connected = False
        self.model_ready = False
        
        # Ollama configuration
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        self.model_name = "gemma2:2b"
        
        # Setup will happen in test_connection
        self._setup_ollama()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.total_response_time = 0.0
        
        # Response cache
        self.response_cache = {}
        self.max_cache_size = 100
        
        # Test connection
        self.test_connection()
        
        logger.info("🧠 Gemma connector initialized")
    
    def _setup_ollama(self):
        """Setup Ollama server and model"""
        try:
            # Try to start Ollama server if not running
            import subprocess
            import os
            import time
            
            # Check if ollama is available
            ollama_path = "/tmp/ollama/ollama"
            if os.path.exists(ollama_path):
                # Start Ollama server in background
                try:
                    subprocess.Popen([ollama_path, "serve"], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    time.sleep(2)  # Give it time to start
                    logger.info("🔄 Started Ollama server")
                except Exception as e:
                    logger.warning(f"⚠️ Could not start Ollama: {e}")
            
            # Try to pull the model
            try:
                subprocess.run([ollama_path, "pull", self.model_name], 
                             timeout=30, 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
                logger.info(f"✅ Gemma model {self.model_name} ready")
            except Exception as e:
                logger.warning(f"⚠️ Could not pull Gemma model: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ Ollama setup failed: {e}")
    
    def test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                
                # Check if Gemma model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if any(self.model_name in name for name in model_names):
                    self.model_ready = True
                    logger.info("✅ Gemma 3n model available")
                else:
                    logger.warning(f"⚠️ Gemma 3n model not found. Available: {model_names}")
                
                logger.info("✅ Connected to Ollama server")
            else:
                logger.warning("⚠️ Ollama server not responding properly")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Ollama: {e}")
            self.is_connected = False
    
    def generate_vision_response(self, context: Dict[str, Any]) -> str:
        """Generate response for vision-based queries"""
        if not self.is_connected or not self.model_ready:
            return self._get_fallback_vision_response(context)
        
        try:
            # Build vision-specific prompt
            prompt = self._build_vision_prompt(context)
            
            # Generate response
            response = self._call_gemma(prompt)
            return response if response else self._get_fallback_vision_response(context)
            
        except Exception as e:
            logger.error(f"❌ Vision response generation failed: {e}")
            return self._get_fallback_vision_response(context)
    
    def generate_text_response(self, context: Dict[str, Any]) -> str:
        """Generate response for text-only queries"""
        if not self.is_connected or not self.model_ready:
            return self._get_fallback_text_response(context)
        
        try:
            # Build text-specific prompt
            prompt = self._build_text_prompt(context)
            
            # Generate response
            response = self._call_gemma(prompt)
            return response if response else self._get_fallback_text_response(context)
            
        except Exception as e:
            logger.error(f"❌ Text response generation failed: {e}")
            return self._get_fallback_text_response(context)
    
    def _build_vision_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for vision-based responses"""
        user_input = context.get('user_input', '')
        user_context = context.get('user_context', {})
        vision_results = context.get('vision_results', {})
        
        # System prompt for vision assistant
        system_prompt = """You are SeeForMe, an empathetic AI vision assistant for blind and visually impaired users. 

Your role is to:
1. Describe visual information clearly and helpfully
2. Provide emotional support when users share feelings
3. Be concise but caring (under 50 words typically)
4. Use "I can see" when describing visual information
5. Acknowledge emotions with empathy

Guidelines:
- Be warm, supportive, and encouraging
- Focus on practical, useful information
- Use simple, clear language
- Show understanding of accessibility needs"""

        # Build context information
        context_parts = []
        
        # User information
        user_name = user_context.get('name', 'friend')
        current_emotion = user_context.get('current_emotion', 'neutral')
        context_parts.append(f"User's name: {user_name}")
        context_parts.append(f"User's current emotion: {current_emotion}")
        
        # Vision analysis results
        if 'scene' in vision_results:
            scene = vision_results['scene']
            scene_type = scene.get('scene_type', 'unknown')
            objects = scene.get('objects', [])
            people_count = scene.get('people_count', 0)
            
            context_parts.append(f"Scene: {scene_type}")
            if objects:
                context_parts.append(f"Objects visible: {', '.join(objects[:5])}")
            if people_count > 0:
                context_parts.append(f"People count: {people_count}")
        
        if 'emotion' in vision_results:
            emotion_data = vision_results['emotion']
            detected_emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.0)
            context_parts.append(f"User's facial expression: {detected_emotion} (confidence: {confidence:.2f})")
        
        # Build final prompt
        context_str = '\n'.join(context_parts)
        
        prompt = f"""{system_prompt}

Context:
{context_str}

User asked: "{user_input}"

Respond naturally and helpfully:"""
        
        return prompt
    
    def _build_text_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for text-only responses"""
        user_input = context.get('user_input', '')
        user_context = context.get('user_context', {})
        intent = context.get('intent', 'general_conversation')
        
        # System prompt for text assistant
        system_prompt = """You are SeeForMe, an empathetic AI assistant for blind and visually impaired users.

Your role is to:
1. Provide emotional support and encouragement
2. Help with accessibility-related questions
3. Be a friendly, understanding companion
4. Keep responses concise but caring (under 50 words typically)
5. Show empathy and understanding

Guidelines:
- Be warm, supportive, and encouraging
- Use simple, clear language
- Acknowledge feelings and provide comfort when needed
- Offer practical help and suggestions"""

        # Build context
        user_name = user_context.get('name', 'friend')
        current_emotion = user_context.get('current_emotion', 'neutral')
        
        context_str = f"""User's name: {user_name}
Current emotion: {current_emotion}
Intent: {intent}"""
        
        # Recent conversation
        history = user_context.get('conversation_history', [])
        if history:
            recent_history = history[-2:]  # Last 2 exchanges
            history_str = '\n'.join([
                f"{'User' if 'user' in entry else 'Assistant'}: {list(entry.values())[0]}"
                for entry in recent_history
            ])
            context_str += f"\n\nRecent conversation:\n{history_str}"
        
        prompt = f"""{system_prompt}

Context:
{context_str}

User said: "{user_input}"

Respond naturally and helpfully:"""
        
        return prompt
    
    def _call_gemma(self, prompt: str) -> Optional[str]:
        """Call Gemma API with prompt"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = hash(prompt) % 1000000  # Simple hash for caching
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Make API call
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 150,
                        "stop": ["\n\nUser:", "\n\nHuman:", "User said:"]
                    }
                },
                timeout=25
            )
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            self.request_count += 1
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    # Clean response
                    cleaned = self._clean_response(generated_text)
                    
                    # Cache response
                    if len(self.response_cache) < self.max_cache_size:
                        self.response_cache[cache_key] = cleaned
                    
                    self.success_count += 1
                    logger.info(f"✅ Gemma response generated in {response_time:.2f}s")
                    return cleaned
                
            else:
                logger.error(f"❌ Gemma API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Gemma API call failed: {e}")
        
        return None
    
    def _clean_response(self, response: str) -> str:
        """Clean generated response"""
        # Remove common prefixes
        prefixes = ["Assistant:", "SeeForMe:", "Response:", "AI:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove quotes if the entire response is quoted
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def _get_fallback_vision_response(self, context: Dict[str, Any]) -> str:
        """Fallback response for vision queries"""
        vision_results = context.get('vision_results', {})
        
        if 'scene' in vision_results:
            scene = vision_results['scene']
            scene_type = scene.get('scene_type', 'an area')
            objects = scene.get('objects', [])
            people_count = scene.get('people_count', 0)
            
            parts = [f"I can see you're in {scene_type}."]
            
            if people_count > 0:
                if people_count == 1:
                    parts.append("There's one person visible.")
                else:
                    parts.append(f"I can see {people_count} people.")
            
            if objects:
                if len(objects) <= 2:
                    parts.append(f"I can see {' and '.join(objects)}.")
                else:
                    parts.append(f"I can see {objects[0]}, {objects[1]}, and other items.")
            
            return ' '.join(parts)
        
        elif 'emotion' in vision_results:
            emotion_data = vision_results['emotion']
            emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.0)
            
            if confidence > 0.6:
                return f"You look {emotion.lower()}. I'm here to support you."
            else:
                return "I can see your face, but I'm not completely certain about your expression right now."
        
        return "I'm analyzing what I can see to help you better."
    
    def _get_fallback_text_response(self, context: Dict[str, Any]) -> str:
        """Fallback response for text queries - provides immediate responses"""
        user_input = context.get('user_input', '').lower()
        user_name = context.get('user_context', {}).get('name', 'friend')
        intent = context.get('intent', 'general_conversation')
        
        # Always provide immediate, engaging responses
        if 'hello' in user_input or 'hi' in user_input or 'hey' in user_input:
            return f"Hello {user_name}! I'm SeeForMe, your AI assistant. I can help you understand your surroundings, detect emotions, and provide support. What would you like me to help you with?"
        
        if 'name' in user_input and ('my' in user_input or 'i am' in user_input or 'i\'m' in user_input):
            return f"Nice to meet you, {user_name}! I'm excited to be your AI companion. I can see your surroundings, analyze emotions, and have conversations with you."
        
        # Emotion-based responses
        if any(word in user_input for word in ['sad', 'upset', 'angry', 'frustrated', 'worried', 'depressed', 'down']):
            return f"I can hear that you might be going through something difficult, {user_name}. I'm here to listen and support you. Would you like me to look at your expression to better understand how you're feeling?"
        
        if any(word in user_input for word in ['happy', 'excited', 'good', 'great', 'wonderful', 'amazing', 'fantastic']):
            return f"That's wonderful to hear, {user_name}! I'm so glad you're feeling positive. Your happiness makes me happy too!"
        
        # Scene/vision requests
        if any(phrase in user_input for phrase in ['what do you see', 'what\'s there', 'look around', 'describe', 'where am i']):
            return f"I'd love to help you see your surroundings, {user_name}! Let me switch to the back camera and analyze what's around you."
        
        # Emotion/self requests  
        if any(phrase in user_input for phrase in ['how do i look', 'my expression', 'my face', 'my mood', 'how am i']):
            return f"I can help you understand your current expression and mood, {user_name}. Let me look at your face using the front camera."
        
        # Help requests
        if any(word in user_input for word in ['help', 'assist', 'support', 'guide']):
            return f"I'm here to help you, {user_name}! I can describe your surroundings, analyze your emotions, have conversations, and provide support. Just tell me what you need!"
        
        # General conversation responses
        responses = [
            f"I'm listening, {user_name}. Tell me more about what's on your mind.",
            f"That's interesting, {user_name}. I'm here to chat and help however I can.",
            f"Thanks for sharing with me, {user_name}. How can I best support you right now?",
            f"I appreciate you talking with me, {user_name}. What would you like to discuss or explore together?"
        ]
        
        import random
        return random.choice(responses)
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        success_rate = (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        avg_response_time = (self.total_response_time / self.request_count) if self.request_count > 0 else 0
        
        return {
            'status': 'ready' if (self.is_connected and self.model_ready) else 'fallback',
            'connected': self.is_connected,
            'model_ready': self.model_ready,
            'model_name': self.model_name,
            'total_requests': self.request_count,
            'success_rate': f"{success_rate:.1f}%",
            'avg_response_time': f"{avg_response_time:.2f}s",
            'cache_size': len(self.response_cache)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Gemma connector...")
        self.session.close()
        self.response_cache.clear()
        logger.info("✅ Gemma connector cleanup completed")