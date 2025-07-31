#!/usr/bin/env python3
"""
TTS Handler - Multi-language text-to-speech
Offline English via pyttsx3, online fallback for Hindi/Gujarati via gTTS
"""

import logging
import os
import threading
import time
import tempfile
from typing import Optional, Dict, Any
import pygame

logger = logging.getLogger(__name__)

class TTSHandler:
    """Multi-language text-to-speech handler"""
    
    def __init__(self):
        self.is_initialized = False
        self.pyttsx3_engine = None
        self.pygame_initialized = False
        
        # Language support
        self.supported_languages = ['en', 'hi', 'gu']
        self.offline_languages = ['en']  # Languages with offline support
        
        # Audio settings
        self.speech_rate = 150
        self.volume = 0.9
        
        # Threading
        self.speech_lock = threading.Lock()
        self.is_speaking = False
        
        # Initialize components
        self.initialize_offline_tts()
        self.initialize_pygame()
        
        logger.info("🔊 TTS handler initialized")
    
    def initialize_offline_tts(self):
        """Initialize pyttsx3 for offline English TTS"""
        try:
            import pyttsx3
            
            self.pyttsx3_engine = pyttsx3.init()
            
            # Configure voice properties
            self.pyttsx3_engine.setProperty('rate', self.speech_rate)
            self.pyttsx3_engine.setProperty('volume', self.volume)
            
            # Try to set a natural voice
            voices = self.pyttsx3_engine.getProperty('voices')
            if voices:
                # Prefer female voices for accessibility
                female_voices = [v for v in voices if 'female' in v.name.lower() or 'woman' in v.name.lower()]
                if female_voices:
                    self.pyttsx3_engine.setProperty('voice', female_voices[0].id)
                else:
                    self.pyttsx3_engine.setProperty('voice', voices[0].id)
            
            self.is_initialized = True
            logger.info("✅ Offline TTS (pyttsx3) initialized")
            
        except ImportError:
            logger.warning("⚠️ pyttsx3 not available, using fallback TTS")
        except Exception as e:
            logger.error(f"❌ Offline TTS initialization failed: {e}")
    
    def initialize_pygame(self):
        """Initialize pygame for audio playback"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.pygame_initialized = True
            logger.info("✅ Pygame audio initialized")
        except Exception as e:
            logger.error(f"❌ Pygame initialization failed: {e}")
    
    def speak(self, text: str, language: str = 'en', blocking: bool = False):
        """Speak text in specified language"""
        if not text or not text.strip():
            return
        
        # Clean text
        text = text.strip()
        
        if language in self.offline_languages and self.pyttsx3_engine:
            self._speak_offline(text, blocking)
        else:
            self._speak_online(text, language, blocking)
    
    def _speak_offline(self, text: str, blocking: bool = False):
        """Speak using offline pyttsx3"""
        with self.speech_lock:
            try:
                self.is_speaking = True
                
                if blocking:
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                else:
                    # Non-blocking speech
                    def speak_thread():
                        try:
                            self.pyttsx3_engine.say(text)
                            self.pyttsx3_engine.runAndWait()
                        except Exception as e:
                            logger.error(f"❌ Offline speech failed: {e}")
                        finally:
                            self.is_speaking = False
                    
                    thread = threading.Thread(target=speak_thread, daemon=True)
                    thread.start()
                
                logger.info(f"🗣️ Speaking offline: '{text[:50]}...'")
                
            except Exception as e:
                logger.error(f"❌ Offline speech failed: {e}")
                self.is_speaking = False
    
    def _speak_online(self, text: str, language: str = 'en', blocking: bool = False):
        """Speak using online gTTS"""
        if not blocking:
            # Non-blocking speech
            thread = threading.Thread(
                target=self._speak_online_blocking, 
                args=(text, language), 
                daemon=True
            )
            thread.start()
        else:
            self._speak_online_blocking(text, language)
    
    def _speak_online_blocking(self, text: str, language: str = 'en'):
        """Blocking online speech using gTTS"""
        with self.speech_lock:
            try:
                from gtts import gTTS
                
                self.is_speaking = True
                
                # Map language codes
                lang_map = {
                    'en': 'en',
                    'hi': 'hi',
                    'gu': 'gu'
                }
                gtts_lang = lang_map.get(language, 'en')
                
                # Generate speech
                tts = gTTS(text=text, lang=gtts_lang, slow=False)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    temp_path = tmp_file.name
                    tts.save(temp_path)
                
                # Play audio using pygame
                if self.pygame_initialized:
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                logger.info(f"🗣️ Speaking online ({language}): '{text[:50]}...'")
                
            except ImportError:
                logger.warning("⚠️ gTTS not available, using fallback")
                self._fallback_speak(text)
            except Exception as e:
                logger.error(f"❌ Online speech failed: {e}")
                self._fallback_speak(text)
            finally:
                self.is_speaking = False
    
    def _fallback_speak(self, text: str):
        """Fallback speech method"""
        # Try offline TTS as fallback
        if self.pyttsx3_engine:
            self._speak_offline(text, blocking=False)
        else:
            logger.warning(f"⚠️ Cannot speak: '{text[:50]}...' (no TTS available)")
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            if self.pyttsx3_engine:
                self.pyttsx3_engine.stop()
            
            if self.pygame_initialized:
                pygame.mixer.music.stop()
            
            self.is_speaking = False
            logger.info("🛑 Speech stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop speech: {e}")
    
    def set_speech_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        if self.pyttsx3_engine:
            self.speech_rate = max(50, min(300, rate))
            self.pyttsx3_engine.setProperty('rate', self.speech_rate)
            logger.info(f"🎚️ Speech rate set to {self.speech_rate} WPM")
    
    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        if self.pyttsx3_engine:
            self.volume = max(0.0, min(1.0, volume))
            self.pyttsx3_engine.setProperty('volume', self.volume)
            logger.info(f"🔊 Volume set to {self.volume}")
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        voices = []
        
        if self.pyttsx3_engine:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'language': getattr(voice, 'languages', ['en'])[0] if hasattr(voice, 'languages') else 'en',
                        'gender': 'female' if 'female' in voice.name.lower() else 'male',
                        'engine': 'pyttsx3'
                    })
            except Exception as e:
                logger.error(f"❌ Failed to get voices: {e}")
        
        return voices
    
    def set_voice(self, voice_id: str):
        """Set specific voice by ID"""
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.setProperty('voice', voice_id)
                logger.info(f"🎭 Voice set to: {voice_id}")
            except Exception as e:
                logger.error(f"❌ Failed to set voice: {e}")
    
    def test_speech(self, language: str = 'en'):
        """Test speech functionality"""
        test_messages = {
            'en': "Hello! This is a test of the speech system. I am SeeForMe, your vision assistant.",
            'hi': "नमस्ते! यह स्पीच सिस्टम का टेस्ट है। मैं सीफॉरमी हूं, आपका विज़न असिस्टेंट।",
            'gu': "નમસ્તે! આ સ્પીચ સિસ્ટમનો ટેસ્ટ છે. હું સીફોરમી છું, તમારો વિઝન આસિસ્ટન્ટ."
        }
        
        test_text = test_messages.get(language, test_messages['en'])
        self.speak(test_text, language, blocking=True)
        
        return {
            'language': language,
            'text': test_text,
            'method': 'offline' if language in self.offline_languages else 'online',
            'success': True
        }
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported"""
        return language in self.supported_languages
    
    def get_status(self) -> Dict[str, Any]:
        """Get current TTS status"""
        return {
            'status': 'ready' if self.is_initialized else 'limited',
            'speaking': self.is_speaking,
            'offline_engine': self.pyttsx3_engine is not None,
            'pygame_audio': self.pygame_initialized,
            'supported_languages': self.supported_languages,
            'offline_languages': self.offline_languages,
            'speech_rate': self.speech_rate,
            'volume': self.volume,
            'available_voices': len(self.get_available_voices())
        }
    
    def cleanup(self):
        """Cleanup TTS resources"""
        logger.info("🧹 Cleaning up TTS handler...")
        
        # Stop any ongoing speech
        self.stop_speaking()
        
        # Cleanup pyttsx3
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except:
                pass
            self.pyttsx3_engine = None
        
        # Cleanup pygame
        if self.pygame_initialized:
            try:
                pygame.mixer.quit()
            except:
                pass
            self.pygame_initialized = False
        
        self.is_initialized = False
        logger.info("✅ TTS cleanup completed")