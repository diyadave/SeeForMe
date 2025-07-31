#!/usr/bin/env python3
"""
Simple Gemma 3n Agent - Direct AI integration without complex initialization
Fast response system for emotional intelligence
"""

import logging
import requests
import json
import subprocess
import os
import time

logger = logging.getLogger(__name__)

class SimpleGemmaAgent:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "gemma:3b"
        self.is_connected = False
        self.session = requests.Session()
        self.session.timeout = 3
        
        # Try to ensure Ollama is running
        self._ensure_ollama()
        
    def _ensure_ollama(self):
        """Ensure Ollama server is running with Gemma 3b"""
        try:
            # Check if already running
            response = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if any('gemma:3b' in str(model) for model in models):
                    self.is_connected = True
                    logger.info("✅ Gemma 3b already running")
                    return
        except:
            pass
            
        # Try to start Ollama and pull model
        try:
            # Start ollama serve in background
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(3)
            
            # Pull Gemma 3b model
            subprocess.run(['ollama', 'pull', 'gemma:3b'], 
                         timeout=60, capture_output=True)
            
            # Test connection again
            response = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if any('gemma:3b' in str(model) for model in models):
                    self.is_connected = True
                    logger.info("✅ Gemma 3b connected and ready")
                    
        except Exception as e:
            logger.warning(f"⚠️ Ollama setup failed: {e}")
    
    def get_response(self, user_input, user_name="", emotion="neutral"):
        """Get fast AI response from Gemma 3b"""
        if not self.is_connected:
            return f"Hello {user_name}! I heard you say '{user_input}'. I'm your AI companion, here to support you emotionally."
            
        try:
            # Craft empathetic prompt for emotional intelligence
            prompt = f"""You are an emotionally intelligent AI companion for a visually impaired person named {user_name}. 
They just said: "{user_input}"
Their current emotion seems: {emotion}

Respond with warmth, empathy, and emotional intelligence. Keep it conversational and supportive, like a caring friend.
Be helpful but not overly clinical. Show you understand their feelings."""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100
                }
            }
            
            response = self.session.post(f"{self.base_url}/api/generate",
                                       json=payload, timeout=8)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                
                if ai_response:
                    logger.info("✅ Gemma 3b AI response generated")
                    return ai_response
                    
        except Exception as e:
            logger.error(f"❌ Gemma generation error: {e}")
            
        # Intelligent fallback with user context
        return f"Hi {user_name}! I heard you say '{user_input}'. I'm your AI companion and I'm here to listen and support you through whatever you're feeling."

# Global instance
simple_agent = SimpleGemmaAgent()