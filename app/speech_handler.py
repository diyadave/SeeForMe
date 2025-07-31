#!/usr/bin/env python3
"""
Speech Handler - Offline voice recognition using Vosk
Supports English, Hindi, and Gujarati with auto-detection
"""

import logging
import threading
import time
import json
import queue
import os
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

class SpeechHandler:
    """Offline speech recognition using Vosk models"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.is_listening = False
        self.is_initialized = False
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 4096
        
        # Language support
        self.supported_languages = ['en', 'hi', 'gu']
        self.current_language = 'en'
        
        # Vosk components
        self.vosk_model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        
        # Threading
        self.audio_thread = None
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Fallback to browser speech recognition
        self.use_browser_fallback = True
        
        logger.info("🎤 Speech handler initialized (browser fallback mode)")
    
    def initialize_vosk(self):
        """Initialize Vosk speech recognition (if available)"""
        try:
            import vosk
            import pyaudio
            
            # Try to load Vosk model
            model_path = self._get_vosk_model_path()
            if model_path and os.path.exists(model_path):
                self.vosk_model = vosk.Model(model_path)
                self.recognizer = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
                self.use_browser_fallback = False
                self.is_initialized = True
                logger.info("✅ Vosk speech recognition initialized")
            else:
                logger.warning("⚠️ Vosk model not found, using browser fallback")
                
        except ImportError:
            logger.warning("⚠️ Vosk not available, using browser fallback")
        except Exception as e:
            logger.error(f"❌ Vosk initialization failed: {e}")
    
    def _get_vosk_model_path(self) -> Optional[str]:
        """Get path to Vosk model for current language"""
        model_paths = {
            'en': 'models/vosk-model-en-us-0.22',
            'hi': 'models/vosk-model-hi-0.22',
            'gu': 'models/vosk-model-gu-0.22'
        }
        return model_paths.get(self.current_language)
    
    def start_listening(self):
        """Start voice recognition"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.stop_event.clear()
        
        if self.use_browser_fallback:
            # Browser-based speech recognition handled in frontend
            logger.info("🎤 Browser speech recognition active")
            if self.callback:
                # Simulate ready state for browser fallback
                threading.Thread(target=self._browser_fallback_status, daemon=True).start()
        else:
            # Start Vosk-based recognition
            self._start_vosk_recognition()
    
    def _browser_fallback_status(self):
        """Send status for browser fallback mode"""
        time.sleep(0.5)  # Brief delay
        # The actual recognition happens in browser, this just signals readiness
    
    def _start_vosk_recognition(self):
        """Start Vosk-based voice recognition"""
        try:
            import pyaudio
            
            # Initialize audio stream
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("🎤 Vosk speech recognition started")
            
            # Audio processing loop
            while self.is_listening and not self.stop_event.is_set():
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    if self.recognizer and self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()
                        
                        if text and self.callback:
                            confidence = result.get('confidence', 1.0)
                            self.callback(text, self.current_language, confidence)
                    
                except Exception as e:
                    logger.error(f"❌ Audio processing error: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
        except Exception as e:
            logger.error(f"❌ Vosk recognition failed: {e}")
    
    def stop_listening(self):
        """Stop voice recognition"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        self.stop_event.set()
        
        logger.info("🛑 Speech recognition stopped")
    
    def set_language(self, language: str):
        """Set recognition language"""
        if language in self.supported_languages:
            self.current_language = language
            logger.info(f"🌐 Language set to: {language}")
            
            # Reinitialize if needed
            if not self.use_browser_fallback:
                self.initialize_vosk()
    
    def detect_language(self, text: str) -> str:
        """Detect language from text (simple heuristic)"""
        # Simple language detection based on character sets
        if any(ord(char) > 2304 and ord(char) < 2431 for char in text):  # Devanagari
            return 'hi'
        elif any(ord(char) > 2688 and ord(char) < 2815 for char in text):  # Gujarati
            return 'gu'
        else:
            return 'en'
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'status': 'listening' if self.is_listening else 'ready',
            'initialized': self.is_initialized,
            'language': self.current_language,
            'supported_languages': self.supported_languages,
            'mode': 'browser' if self.use_browser_fallback else 'vosk',
            'active': self.is_listening
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up speech handler...")
        self.stop_listening()
    
    def process_browser_input(self, text: str, language: str = 'en', confidence: float = 1.0):
        """Process input from browser speech recognition"""
        if self.callback and text.strip():
            # Auto-detect language if not specified
            if language == 'auto':
                language = self.detect_language(text)
            
            self.callback(text, language, confidence)
            logger.info(f"🗣️ Browser speech processed: '{text}' ({language})")