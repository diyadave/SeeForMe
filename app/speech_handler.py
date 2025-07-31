#!/usr/bin/env python3
"""
Advanced Speech Recognition using Vosk
Multi-language support (English/Hindi/Gujarati) with auto-detection
"""

import os
import json
import time
import logging
import threading
import queue
import wave
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
import sys

try:
    import vosk  # pip install vosk
except ImportError:
    vosk = None
    logging.error("❌ Vosk not installed. Run: pip install vosk")

try:
    import pyaudio
except ImportError:
    pyaudio = None
    logging.error("❌ PyAudio not installed. Run: pip install pyaudio")

logger = logging.getLogger(__name__)


class SpeechHandler:
    """Advanced speech recognition using Vosk with multi-language support"""
    
    def __init__(self, model_path: str = "models"):
        logger.info("🎤 Initializing Vosk Speech Recognizer...")
        
        self.vosk_available = bool(vosk)
        if not vosk:
            logger.warning("⚠️ Vosk not available, using browser fallback")
        
        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 4096
        self.audio_format = pyaudio.paInt16 if pyaudio else None
        
        # Model paths
        self.model_path = Path(model_path)
        self.models = {}
        self.current_model = None
        self.current_language = 'en'
        
        # Recognition state
        self.is_listening = False
        self.is_initialized = False
        self.callback = None
        
        # Audio components
        self.audio = None
        self.stream = None
        
        # Threading
        self.recognition_thread = None
        self.audio_queue = queue.Queue()
        
        # Performance tracking
        self.recognition_count = 0
        self.successful_recognitions = 0
        self.average_confidence = 0.0
        
        # Language detection patterns
        self.language_patterns = {
            'en': [
                'the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                'will', 'would', 'could', 'should', 'can', 'may', 'might',
                'hello', 'hi', 'hey', 'good', 'morning', 'evening', 'night',
                'thank', 'you', 'please', 'sorry', 'yes', 'no', 'okay'
            ],
            'hi': [
                'hai', 'hain', 'hun', 'hoon', 'tha', 'thi', 'the', 'kya', 'koi',
                'main', 'mai', 'mera', 'meri', 'mere', 'aap', 'aapka', 'aapki',
                'namaste', 'namaskar', 'dhanyawad', 'maaf', 'haan', 'nahi', 'theek'
            ],
            'gu': [
                'che', 'chhe', 'hato', 'hati', 'shu', 'koi', 'hu', 'maru', 'mari',
                'mare', 'tame', 'tamaru', 'tamari', 'namaste', 'namaskar',
                'dhanyawad', 'maaf', 'haan', 'naa', 'saras'
            ]
        }
        
        # Initialize
        self.initialize()
        
        logger.info("✅ Speech Handler initialized")
    
    def initialize(self):
        """Initialize speech recognition"""
        try:
            if self.vosk_available:
                self.initialize_models()
                self.initialize_audio()
            else:
                logger.info("🎤 Speech handler initialized (browser fallback mode)")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Speech handler initialization failed: {e}")
            self.is_initialized = False
    
    def initialize_models(self):
        """Initialize Vosk models for different languages"""
        if not self.vosk_available:
            return
            
        logger.info("📚 Loading Vosk models...")
        
        # Expected model directories
        model_dirs = {
            'en': self.model_path / 'vosk-model-en-us-0.22',
            'hi': self.model_path / 'vosk-model-hi-0.22',
            'gu': self.model_path / 'vosk-model-small-gujarati-0.4'
        }
        
        # Also check for alternative model names
        if not any(model_dir.exists() for model_dir in model_dirs.values()):
            # Try alternative patterns
            for lang in ['en', 'hi', 'gu']:
                for model_dir in self.model_path.glob(f'*{lang}*'):
                    if model_dir.is_dir():
                        model_dirs[lang] = model_dir
                        break
        
        # Load available models
        for lang, model_dir in model_dirs.items():
            if model_dir.exists():
                try:
                    logger.info(f"📖 Loading {lang} model from {model_dir}")
                    model = vosk.Model(str(model_dir))
                    self.models[lang] = model
                    logger.info(f"✅ {lang} model loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load {lang} model: {e}")
            else:
                logger.warning(f"⚠️ Model not found: {model_dir}")
        
        if not self.models:
            logger.warning("⚠️ No Vosk models found, falling back to browser recognition")
            return
        
        # Set default model
        if 'en' in self.models:
            self.current_model = self.models['en']
            self.current_language = 'en'
        else:
            # Use first available model
            first_lang = list(self.models.keys())[0]
            self.current_model = self.models[first_lang]
            self.current_language = first_lang
        
        logger.info(f"🎯 Default model set to: {self.current_language}")
    
    def initialize_audio(self):
        """Initialize PyAudio for microphone input"""
        if not self.vosk_available or not pyaudio:
            logger.warning("⚠️ PyAudio not available, using browser fallback")
            return
            
        logger.info("🎙️ Initializing audio system...")
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Test audio stream
            test_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            test_stream.close()
            
            logger.info("✅ Audio system initialized")
            
        except Exception as e:
            logger.error(f"❌ Audio initialization failed: {e}")
            self.vosk_available = False
    
    def set_callback(self, callback: Callable[[str, str, float], None]):
        """Set callback for speech recognition results"""
        self.callback = callback
    
    def start_listening(self) -> bool:
        """Start continuous speech recognition"""
        if self.is_listening:
            logger.info("🎤 Already listening")
            return True
        
        if not self.is_initialized:
            logger.error("❌ Speech recognizer not initialized")
            return False
        
        if not self.vosk_available or not self.models:
            logger.info("🎤 Using browser-based speech recognition")
            self.is_listening = True
            return True
        
        logger.info("🎤 Starting Vosk speech recognition...")
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            self.is_listening = True
            
            # Start recognition thread
            self.recognition_thread = threading.Thread(target=self.recognition_worker, daemon=True)
            self.recognition_thread.start()
            
            logger.info("✅ Vosk speech recognition started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start listening: {e}")
            return False
    
    def stop_listening(self):
        """Stop speech recognition"""
        if not self.is_listening:
            return
        
        logger.info("⏹️ Stopping speech recognition...")
        
        self.is_listening = False
        
        # Stop and close audio stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        # Wait for recognition thread to finish
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=2)
        
        logger.info("✅ Speech recognition stopped")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous audio capture"""
        if self.is_listening:
            # Add audio data to queue
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                # Remove oldest audio if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(in_data)
                except queue.Empty:
                    pass
        
        return (None, pyaudio.paContinue)
    
    def recognition_worker(self):
        """Background worker for speech recognition"""
        logger.info("🤖 Recognition worker started")
        
        # Create recognizer for current model
        rec = vosk.KaldiRecognizer(self.current_model, self.sample_rate)
        rec.SetWords(True)  # Enable word-level timestamps
        
        # Audio buffer
        audio_buffer = b''
        buffer_size = self.chunk_size * 4  # Buffer multiple chunks
        
        while self.is_listening:
            try:
                # Get audio data from queue
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Add to buffer
                audio_buffer += audio_data
                
                # Process when buffer is large enough
                if len(audio_buffer) >= buffer_size:
                    # Feed audio to recognizer
                    if rec.AcceptWaveform(audio_buffer):
                        # Final result
                        result = json.loads(rec.Result())
                        if result.get('text'):
                            self.process_recognition_result(result)
                    else:
                        # Partial result
                        partial = json.loads(rec.PartialResult())
                        if partial.get('partial'):
                            logger.debug(f"🎤 Partial: {partial['partial']}")
                    
                    # Clear buffer
                    audio_buffer = b''
                
            except Exception as e:
                logger.error(f"❌ Recognition worker error: {e}")
                time.sleep(0.1)
        
        # Final processing
        try:
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text'):
                self.process_recognition_result(final_result)
        except:
            pass
        
        logger.info("🤖 Recognition worker stopped")
    
    def process_recognition_result(self, result: Dict[str, Any]):
        """Process speech recognition result"""
        text = result.get('text', '').strip()
        if not text:
            return
        
        # Calculate confidence (Vosk doesn't provide confidence directly)
        confidence = min(len(text.split()) / 10.0, 1.0)  # Simple heuristic
        
        # Detect language
        detected_lang = self.detect_language(text)
        
        logger.info(f"🗣️ Recognized [{detected_lang}]: '{text}' (confidence: {confidence:.2f})")
        
        # Update statistics
        self.recognition_count += 1
        self.successful_recognitions += 1
        self.average_confidence = (self.average_confidence * (self.recognition_count - 1) + confidence) / self.recognition_count
        
        # Call callback
        if self.callback:
            self.callback(text, detected_lang, confidence)
    
    def detect_language(self, text: str) -> str:
        """Detect language from recognized text"""
        if not text:
            return self.current_language
        
        text_lower = text.lower()
        word_scores = {'en': 0, 'hi': 0, 'gu': 0}
        
        # Count language-specific words
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    word_scores[lang] += 1
        
        # Find language with highest score
        detected_lang = max(word_scores.items(), key=lambda x: x[1])
        
        if detected_lang[1] > 0:  # At least one pattern matched
            return detected_lang[0]
        
        # Fallback to current language
        return self.current_language
    
    def process_speech_result(self, text: str, language: str = 'en', confidence: float = 1.0):
        """Process speech recognition result from browser (fallback)"""
        logger.info(f"🗣️ Browser speech [{language}]: '{text}' (confidence: {confidence:.2f})")
        
        if self.callback:
            self.callback(text, language, confidence)
        
        self.recognition_count += 1
        if confidence > 0.5:
            self.successful_recognitions += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'status': 'listening' if self.is_listening else 'ready',
            'initialized': self.is_initialized,
            'vosk_available': self.vosk_available,
            'models_loaded': list(self.models.keys()) if self.vosk_available else [],
            'current_language': self.current_language,
            'recognition_count': self.recognition_count,
            'success_rate': self.successful_recognitions / max(self.recognition_count, 1),
            'average_confidence': self.average_confidence
        }