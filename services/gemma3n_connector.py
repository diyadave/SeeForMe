#!/usr/bin/env python3
"""
Gemma 3n Integration for SeeForMe
Connects to Ollama server running Gemma 3n model for intelligent conversations
"""

import os
import requests
import json
import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class Gemma3nConnector:
    """Enhanced Gemma 3n connector for vision accessibility assistant"""
    
    def __init__(self):
        logger.info("🧠 Initializing Gemma 3n Connector...")
        
        # Ollama configuration
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        self.model_name = "gemma3n:latest"  # Gemma 3n model identifier
        
        # Connection management
        self.session = requests.Session()
        self.session.timeout = 30
        self.is_connected = False
        self.model_ready = False
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.total_response_time = 0.0
        self.response_cache = {}
        self.cache_hits = 0
        
        # Conversation context for vision assistant
        self.conversation_history = []
        self.max_history_length = 10
        
        # Initialize connection
        self.initialize_connection()
        
        logger.info("✅ Gemma 3n Connector initialized")
    
    def initialize_connection(self):
        """Initialize connection to Ollama and check model availability"""
        try:
            # Check Ollama server status
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                logger.info("✅ Connected to Ollama server")
                
                # Check if Gemma model is available
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if any(self.model_name in model for model in available_models):
                    self.model_ready = True
                    logger.info(f"✅ Gemma 3n model '{self.model_name}' is available")
                else:
                    logger.warning(f"⚠️ Gemma 3n model '{self.model_name}' not found. Available models: {available_models}")
                    # Try to pull the model
                    self.pull_model()
            else:
                logger.warning("⚠️ Ollama server not responding")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Failed to connect to Ollama: {e}")
            self.is_connected = False
    
    def pull_model(self):
        """Pull Gemma 3n model if not available"""
        logger.info(f"📥 Pulling Gemma 3n model: {self.model_name}")
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes for model download
            )
            if response.status_code == 200:
                self.model_ready = True
                logger.info("✅ Gemma 3n model pulled successfully")
            else:
                logger.error(f"❌ Failed to pull model: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error pulling model: {e}")
    
    def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate empathetic response using Gemma 3n"""
        if not self.is_connected or not self.model_ready:
            return self.get_fallback_response(user_input, context)
        
        # Check cache first
        cache_key = self.get_cache_key(user_input, context)
        if cache_key in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Build context-aware prompt
            prompt = self.build_accessibility_prompt(user_input, context)
            
            # Make request to Ollama
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 200,
                        "stop": ["\n\nUser:", "\n\nHuman:"]
                    }
                },
                timeout=30
            )
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            self.request_count += 1
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    self.success_count += 1
                    
                    # Clean and process response
                    processed_response = self.process_response(generated_text)
                    
                    # Cache the response
                    self.response_cache[cache_key] = processed_response
                    
                    # Update conversation history
                    self.update_conversation_history(user_input, processed_response)
                    
                    logger.info(f"✅ Generated response in {response_time:.2f}s")
                    return processed_response
                else:
                    logger.warning("⚠️ Empty response from Gemma")
                    return self.get_fallback_response(user_input, context)
            else:
                logger.error(f"❌ Gemma API error: {response.status_code} - {response.text}")
                return self.get_fallback_response(user_input, context)
                
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return self.get_fallback_response(user_input, context)
    
    def build_accessibility_prompt(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Build context-aware prompt for vision accessibility assistant"""
        
        # System prompt for SeeForMe vision assistant
        system_prompt = """You are SeeForMe, an empathetic AI assistant designed specifically for blind and visually impaired users. Your role is to:

1. Provide emotional support and encouragement
2. Help users understand their visual environment through camera analysis
3. Respond with warmth, empathy, and practical guidance
4. Be concise but caring in your responses
5. Always speak in first person as their personal assistant

Key guidelines:
- Use simple, clear language
- Be encouraging and positive
- Acknowledge the user's feelings and needs
- Provide practical, actionable help
- Keep responses under 50 words when possible
- Use "I can see" or "I notice" when describing visual information
"""
        
        # Add context information
        context_info = ""
        if context:
            user_name = context.get('user_name', 'friend')
            current_emotion = context.get('current_emotion', 'neutral')
            scene_info = context.get('scene_info', '')
            language = context.get('language', 'en')
            
            context_info = f"""
Current context:
- User's name: {user_name}
- Detected emotion: {current_emotion}
- Scene: {scene_info}
- Language: {language}
"""
        
        # Add recent conversation history
        history_text = ""
        if self.conversation_history:
            history_text = "\nRecent conversation:\n"
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n"
        
        # Build final prompt
        full_prompt = f"""{system_prompt}
{context_info}
{history_text}

User: {user_input}
        
        return full_prompt
    
    def process_response(self, response: str) -> str:
        """Clean and process Gemma response"""
        # Remove any unwanted prefixes or suffixes
        response = response.strip()
        
        # Remove common AI assistant artifacts
        artifacts = ["Assistant:", "SeeForMe:", "AI:", "Response:"]
        for artifact in artifacts:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Ensure proper sentence ending
        if not response.endswith((".", "!", "?")):
            response += "."
        
        return response
    
    def get_fallback_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Provide fallback response when Gemma is unavailable"""
        fallback_responses = [
            "I'm here to help you. My AI processing is loading - please give me a moment.",
            "I understand you're speaking to me. My advanced conversation system is getting ready.",
            "Thank you for your patience. I'm preparing to give you the best assistance possible.",
            "I'm listening and learning about your needs. My full capabilities will be available shortly."
        ]
        
        import random
        return random.choice(fallback_responses)
    
    def update_conversation_history(self, user_input: str, response: str):
        """Update conversation history for context"""
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": time.time()
        })
        
        # Maintain history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
    
    def get_cache_key(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate cache key for response caching"""
        key_parts = [user_input.lower().strip()]
        if context:
            key_parts.append(str(context.get("current_emotion", "")))
            key_parts.append(str(context.get("scene_info", "")))
        return "|".join(key_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status information"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        success_rate = (
            (self.success_count / self.request_count * 100) 
            if self.request_count > 0 else 0
        )
        
        return {
            "connected": self.is_connected,
            "model_ready": self.model_ready,
            "model_name": self.model_name,
            "total_requests": self.request_count,
            "success_rate": f"{success_rate:.1f}%",
            "avg_response_time": f"{avg_response_time:.2f}s",
            "cache_size": len(self.response_cache),
            "cache_hits": self.cache_hits,
            "conversation_length": len(self.conversation_history)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Gemma 3n connector...")
        self.session.close()
        self.conversation_history.clear()
        self.response_cache.clear()
