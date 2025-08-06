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
import pyaudio
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
import sys

# Vosk import with fallback
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    vosk = None
    VOSK_AVAILABLE = False
    logging.warning("⚠️ Vosk not installed. Speech recognition disabled. Run: pip install vosk")

logger = logging.getLogger(__name__)

class VoskSpeechRecognizer:
    """Advanced speech recognition using Vosk with multi-language support"""
    
    def __init__(self, model_path: str = "models", callback: Optional[Callable] = None):
        logger.info("🎤 Initializing Vosk Speech Recognizer...")
        
        if not VOSK_AVAILABLE:
            logger.warning("⚠️ Vosk not available, speech recognition disabled")
            self.available = False
            return
        
        self.available = True
        self.callback = callback  # Callback for recognized speech
        
        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 4096
        self.audio_format = pyaudio.paInt16
        
        # Model paths
        self.model_path = Path(model_path)
        self.models = {}
        self.current_model = None
        self.current_language = 'en'
        
        # Recognition state
        self.is_listening = False
        self.is_initialized = False
        
        # Audio components
        self.audio = None
        self.stream = None
        
        # Threading
        self.recognition_thread = None
        self.audio_queue = queue.Queue()
        self.stop_flag = threading.Event()
        
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
        
        # Initialize models and audio
        try:
            self.initialize_models()
            self.initialize_audio()
            logger.info("✅ Vosk Speech Recognizer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize speech recognizer: {e}")
            self.available = False
    
    def initialize_models(self):
        """Initialize Vosk models for different languages"""
        if not VOSK_AVAILABLE:
            return
            
        logger.info("📚 Loading Vosk models...")
        
        # Expected model directories
        model_dirs = {
            'en': self.model_path / 'vosk-model-small-en-us-0.15',
            'hi': self.model_path / 'vosk-model-small-hi-0.22',
            'gu': self.model_path / 'vosk-model-small-gujarati-0.4'  # Use English small model for Gujarati
        }
        
        # Also check for any available models
        if self.model_path.exists():
            for model_dir in self.model_path.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('vosk-model'):
                    # Try to determine language from directory name
                    if 'en' in model_dir.name and 'en' not in self.models:
                        model_dirs['en'] = model_dir
                    elif 'hi' in model_dir.name and 'hi' not in self.models:
                        model_dirs['hi'] = model_dir
        
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
            logger.warning("❌ No Vosk models found!")
            # Create a fallback recognizer without language models
            raise Exception("No Vosk models found. Please download models to the models/ directory")
        
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
        logger.info("🎙️ Initializing audio system...")
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # List available audio devices
            logger.info("🔊 Available audio devices:")
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    logger.info(f"   {i}: {device_info['name']} (channels: {device_info['maxInputChannels']})")
            
            # Test audio stream
            test_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            test_stream.close()
            
            self.is_initialized = True
            logger.info("✅ Audio system initialized")
            
        except Exception as e:
            logger.error(f"❌ Audio initialization failed: {e}")
            raise Exception(f"Failed to initialize audio: {e}")
    
    def switch_language(self, language: str) -> bool:
        """Switch to different language model"""
        if not self.available:
            return False
            
        if language not in self.models:
            logger.warning(f"⚠️ Language {language} not available")
            return False
        
        if language == self.current_language:
            logger.info(f"🎯 Already using {language} model")
            return True
        
        logger.info(f"🔄 Switching from {self.current_language} to {language}")
        
        # Stop current recognition if running
        was_listening = self.is_listening
        if was_listening:
            self.stop_listening()
        
        self.current_model = self.models[language]
        self.current_language = language
        
        # Restart recognition if it was running
        if was_listening:
            self.start_listening()
        
        logger.info(f"✅ Switched to {language} model")
        return True
    
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
    
    def start_listening(self) -> bool:
        """Start continuous speech recognition"""
        if not self.available:
            logger.warning("⚠️ Speech recognition not available")
            return False
            
        if self.is_listening:
            logger.info("🎤 Already listening")
            return True
        
        if not self.is_initialized:
            logger.error("❌ Speech recognizer not initialized")
            return False
        
        logger.info("🎤 Starting speech recognition...")
        
        try:
            # Clear stop flag
            self.stop_flag.clear()
            
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
            
            logger.info("✅ Speech recognition started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start listening: {e}")
            return False
    
    def stop_listening(self):
        """Stop speech recognition"""
        if not self.is_listening:
            return
        
        logger.info("⏹️ Stopping speech recognition...")
        
        # Set stop flag
        self.stop_flag.set()
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
        if self.is_listening and not self.stop_flag.is_set():
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
        if not self.current_model:
            logger.error("❌ No model available for recognition")
            return
            
        logger.info("🤖 Recognition worker started")
        
        # Create recognizer for current model
        rec = vosk.KaldiRecognizer(self.current_model, self.sample_rate)
        rec.SetWords(True)  # Enable word-level timestamps
        
        # Audio buffer
        audio_buffer = b''
        buffer_size = self.chunk_size * 4  # Buffer multiple chunks
        
        while self.is_listening and not self.stop_flag.is_set():
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
            if audio_buffer:
                rec.AcceptWaveform(audio_buffer)
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text'):
                self.process_recognition_result(final_result)
        except:
            pass
        
        logger.info("🤖 Recognition worker stopped")
    
    def process_recognition_result(self, result: Dict):
        """Process recognition result"""
        try:
            text = result.get('text', '').strip()
            confidence = result.get('conf', 0.0)
            
            if not text:
                return
            
            self.recognition_count += 1
            
            # Update metrics
            if confidence > 0.5:
                self.successful_recognitions += 1
                self.average_confidence = (
                    (self.average_confidence * (self.successful_recognitions - 1) + confidence) 
                    / self.successful_recognitions
                )
            
            logger.info(f"🎤 Recognized [{self.current_language}]: '{text}' (conf: {confidence:.2f})")
            
            # Auto-detect language
            detected_lang = self.detect_language(text)
            if detected_lang != self.current_language and detected_lang in self.models:
                logger.info(f"🔄 Auto-switching to {detected_lang} based on content")
                # Note: We don't auto-switch to avoid interruption, just log
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback({
                        'text': text,
                        'confidence': confidence,
                        'language': self.current_language,
                        'detected_language': detected_lang,
                        'timestamp': time.time()
                    })
                except Exception as e:
                    logger.error(f"❌ Callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing recognition result: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_listening()
            
            if self.audio:
                self.audio.terminate()
                self.audio = None
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("🧹 Speech recognizer cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def get_status(self) -> Dict:
        """Get current status"""
        success_rate = (
            (self.successful_recognitions / self.recognition_count * 100) 
            if self.recognition_count > 0 else 0
        )
        
        return {
            'available': self.available,
            'is_listening': self.is_listening,
            'current_language': self.current_language,
            'available_languages': list(self.models.keys()) if self.models else [],
            'recognition_count': self.recognition_count,
            'success_rate': f"{success_rate:.1f}%",
            'average_confidence': f"{self.average_confidence:.2f}",
            'queue_size': self.audio_queue.qsize() if hasattr(self.audio_queue, 'qsize') else 0
        }

# Test function
if __name__ == "__main__":
    def test_callback(result):
        print(f"Recognized: {result['text']} ({result['language']})")
    
    recognizer = VoskSpeechRecognizer(callback=test_callback)
    
    if recognizer.available:
        print("Starting recognition... Speak something!")
        recognizer.start_listening()
        
        try:
            time.sleep(10)  # Listen for 10 seconds
        except KeyboardInterrupt:
            pass
        
        recognizer.stop_listening()
        print(f"Final status: {recognizer.get_status()}")
    else:
        print("Speech recognition not available")
    
    recognizer.cleanup()
