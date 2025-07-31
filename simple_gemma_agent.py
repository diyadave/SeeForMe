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
        self.model = "gemma2:2b"
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
                if any('gemma2:2b' in str(model) for model in models):
                    self.is_connected = True
                    logger.info("✅ Gemma2:2b already running")
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
            
            # Pull Gemma 2b model (available and working)
            result = subprocess.run(['ollama', 'pull', 'gemma2:2b'], 
                                  timeout=60, capture_output=True, text=True)
            logger.info(f"📥 Ollama pull result: {result.returncode}")
            
            # Test connection again
            response = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if any('gemma2:2b' in str(model) for model in models):
                    self.is_connected = True
                    logger.info("✅ Gemma2:2b connected and ready")
                    
        except Exception as e:
            logger.warning(f"⚠️ Ollama setup failed: {e}")
    
    def get_response(self, user_input, user_name="", emotion="neutral", context=""):
        """Get fast AI response from Gemma2:2b with emotion and context awareness"""
        # Force connection check every time to ensure we're really connected
        self._ensure_ollama()
        
        if not self.is_connected:
            logger.warning("⚠️ Ollama not connected, using fallback response")
            return f"Hello {user_name}! I heard you say '{user_input}'. I'm your AI companion, here to support you emotionally."
            
        try:
            # Craft deeply empathetic and conversational prompt
            emotion_context = ""
            if emotion != "neutral":
                emotion_context = f"I can see you're feeling {emotion} right now. "
                
            visual_context = ""
            if context:
                visual_context = f"From what I can see around you: {context}. "
            
            prompt = f"""You are a caring, emotionally intelligent AI friend for a visually impaired person named {user_name}. 

{emotion_context}{visual_context}They just said: "{user_input}"

Respond as their supportive friend would - warm, conversational, and emotionally aware. 
- If they seem sad/upset, offer comfort and understanding
- If they're happy, share in their joy  
- Be genuinely interested in them as a person
- Ask follow-up questions to keep the conversation going
- Keep responses natural and friendly, not robotic
- Remember details they share about themselves

Make this feel like talking to a real friend who truly cares."""

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