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
        self.model = "gemma3n:latest"
        self.is_connected = False
        self.session = requests.Session()
        self.session.timeout = 3
        
        # Try to ensure Ollama is running
        self._ensure_ollama()
        
    def _ensure_ollama(self):
        """Ensure Ollama server is running with Gemma 3n - with persistent retry loop"""
        print("🔍 Testing Gemma 3n connection...")
        
        # Keep trying until Gemma is available (like your implementation)
        while not self.is_connected:
            try:
                # Test with actual chat request (more reliable than just checking tags)
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "gemma3n:latest",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False
                    },
                    timeout=120
                )
                
                if response.ok:
                    result = response.json()
                    if "message" in result and "content" in result["message"]:
                        self.is_connected = True
                        logger.info("✅ Gemma 3n connection verified and working")
                        return
                
                print("⏳ Gemma responded but incomplete. Retrying in 20 seconds...")
                
            except Exception as e:
                print(f"⏳ Waiting for Gemma to wake up... retrying in 20 seconds ({e})")
                
                # Try to start Ollama in background if not running
                try:
                    subprocess.Popen(['ollama', 'serve'], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    time.sleep(5)
                    
                    # Try to pull model
                    subprocess.run(['ollama', 'pull', 'gemma3n:latest'], 
                                 timeout=60, capture_output=True)
                except:
                    pass
                
                time.sleep(20)
    
    def get_response(self, user_input, user_name="", emotion="neutral", context=""):
        """Get AI response from Gemma 3n with emotion and context awareness"""
        # Ensure connection with persistent retry (like your code)
        if not self.is_connected:
            self._ensure_ollama()
        
        if not self.is_connected:
            logger.warning("⚠️ Gemma 3n not available, using fallback response")
            # Enhanced fallback for emotional support
            if "my name is" in user_input.lower():
                import re
                match = re.search(r"my name is (\w+)", user_input.lower())
                if match:
                    name = match.group(1).capitalize()
                    return f"Hello {name}! Nice to meet you! I'm so glad you're here. How are you feeling today?"
            return f"Hello {user_name}! I heard you say '{user_input}'. I'm your AI companion, here to support you emotionally."
            
        try:
            # Craft deeply empathetic and conversational prompt
            emotion_context = ""
            if emotion != "neutral":
                emotion_context = f"I can see you're feeling {emotion} right now. "
                
            visual_context = ""
            if context:
                visual_context = f"From what I can see around you: {context}. "
            
            # Special handling for name introductions
            if "my name is" in user_input.lower() or "i'm " in user_input.lower():
                prompt = f"""A visually impaired person just introduced themselves: "{user_input}"

Respond warmly and personally like meeting a new friend:
- Greet them by name enthusiastically  
- Say "nice to meet you [name]" 
- Be genuinely welcoming and friendly
- Ask a caring follow-up question about them
- Keep it natural and conversational, not robotic

Example: "Hello [name]! Nice to meet you! I'm so glad you're here. How are you feeling today?"

Make them feel welcomed and valued as a person."""
            else:
                prompt = f"""You are a caring, emotionally intelligent AI friend for a visually impaired person named {user_name}. 

{emotion_context}{visual_context}They just said: "{user_input}"

Respond as their supportive friend would - warm, conversational, and emotionally aware:
- If they seem sad/upset, offer genuine comfort and understanding
- If they're happy, share in their joy enthusiastically  
- Be genuinely interested in them as a person
- Ask caring follow-up questions to keep conversation flowing
- Keep responses natural and friendly, never robotic
- Remember and reference details they share
- Use their name occasionally to personalize responses

Talk like a real human friend who truly cares about their wellbeing."""

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