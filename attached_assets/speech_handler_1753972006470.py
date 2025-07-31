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
from typing import Dict, Optional, List, Any
import sys
try:
    import vosk  # pip install vosk
except ImportError:
    vosk = None
    logging.error("❌ Vosk not installed. Run: pip install vosk")

logger = logging.getLogger(__name__)


class VoskSpeechRecognizer:
    """Advanced speech recognition using Vosk with multi-language support"""
    
    def __init__(self, model_path: str = "models"):
        logger.info("🎤 Initializing Vosk Speech Recognizer...")
        
        if not vosk:
            raise ImportError("Vosk library not found. Install with: pip install vosk")
        
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
        self.initialize_models()
        self.initialize_audio()
        
        logger.info("✅ Vosk Speech Recognizer initialized")
    
    def initialize_models(self):
        """Initialize Vosk models for different languages"""
        logger.info("📚 Loading Vosk models...")
        
        # Expected model directories
        model_dirs = {
            'en': self.model_path / 'vosk-model-en-us-0.22',
            'hi': self.model_path / 'vosk-model-hi-0.22',
            'gu': self.model_path / 'vosk-model-small-gujarati-0.4'  # Use English small model for Gujarati fallback
        }
        
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
            logger.error("❌ No Vosk models found!")
            # Try to use a simple model if available
            simple_model_path = self.model_path / 'vosk-model-small'
            if simple_model_path.exists():
                try:
                    model = vosk.Model(str(simple_model_path))
                    self.models['en'] = model
                    logger.info("✅ Fallback to simple English model")
                except Exception as e:
                    logger.error(f"❌ Failed to load fallback model: {e}")
                    raise Exception("No usable Vosk models found")
            else:
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
        if language not in self.models:
            logger.warning(f"⚠️ Language {language} not available")
            return False
        
        if language == self.current_language:
            logger.info(f"🎯 Already using {language} model")
            return True
        
        logger.info(f"🔄 Switching from {self.current_language} to {language}")
        
        self.current_model = self.models[language]
        self.current_language = language
        
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
        if self.is_listening:
            logger.info("🎤 Already listening")
            return True
        
        if not self.is_initialized:
            logger.error("❌ Speech recognizer not initialized")
            return False
        
        logger.info("🎤 Starting speech recognition...")
        
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
    
    def process_recognition_result(self, result: Dict) -> Dict:
        """Process recognition result and return formatted data"""
        text = result.get('text', '').strip()
        if not text:
            return None
        
        self.recognition_count += 1
        
        # Calculate confidence (Vosk doesn't provide direct confidence)
        confidence = self.estimate_confidence(result)
        
        # Detect language
        detected_lang = self.detect_language(text)
        
        # Switch model if language changed
        if detected_lang != self.current_language and detected_lang in self.models:
            logger.info(f"🔄 Auto-switching to {detected_lang} based on content")
            self.switch_language(detected_lang)
        
        # Update statistics
        if confidence > 0.5:
            self.successful_recognitions += 1
            self.update_average_confidence(confidence)
        
        # Format result
        formatted_result = {
            'text': text,
            'confidence': confidence,
            'language': detected_lang,
            'timestamp': time.time(),
            'words': result.get('result', [])  # Word-level details if available
        }
        
        logger.info(f"🗣️ Recognized [{detected_lang}]: '{text}' (confidence: {confidence:.2f})")
        
        return formatted_result
    
    def estimate_confidence(self, result: Dict) -> float:
        """Estimate confidence score from Vosk result"""
        # Vosk doesn't provide direct confidence, so we estimate based on:
        # 1. Word-level confidence if available
        # 2. Text length and completeness
        # 3. Recognition context
        
        text = result.get('text', '')
        words = result.get('result', [])
        
        if not text:
            return 0.0
        
        # Use word-level confidence if available
        if words and isinstance(words, list):
            word_confidences = []
            for word_info in words:
                if isinstance(word_info, dict) and 'conf' in word_info:
                    word_confidences.append(word_info['conf'])
            
            if word_confidences:
                return sum(word_confidences) / len(word_confidences)
        
        # Fallback estimation based on text characteristics
        confidence = 0.7  # Base confidence
        
        # Adjust based on text length
        if len(text) > 50:
            confidence += 0.1  # Longer text usually more accurate
        elif len(text) < 5:
            confidence -= 0.2  # Very short text less reliable
        
        # Adjust based on word count
        word_count = len(text.split())
        if word_count >= 3:
            confidence += 0.1
        elif word_count == 1:
            confidence -= 0.1
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def update_average_confidence(self, confidence: float):
        """Update average confidence with exponential moving average"""
        if self.average_confidence == 0:
            self.average_confidence = confidence
        else:
            alpha = 0.3  # Smoothing factor
            self.average_confidence = alpha * confidence + (1 - alpha) * self.average_confidence
    
    def listen_once(self, timeout: float = 5.0) -> Optional[Dict]:
        """Listen for a single speech input with timeout"""
        if not self.is_initialized:
            logger.error("❌ Speech recognizer not initialized")
            return None
        
        logger.info(f"🎤 Listening for {timeout} seconds...")
        
        try:
            # Create recognizer
            rec = vosk.KaldiRecognizer(self.current_model, self.sample_rate)
            rec.SetWords(True)
            
            # Open audio stream
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            start_time = time.time()
            audio_buffer = b''
            
            while time.time() - start_time < timeout:
                # Read audio data
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_buffer += audio_data
                
                # Process audio
                if rec.AcceptWaveform(audio_data):
                    result = json.loads(rec.Result())
                    if result.get('text'):
                        stream.close()
                        return self.process_recognition_result(result)
                
                # Check for partial results
                partial = json.loads(rec.PartialResult())
                if partial.get('partial'):
                    logger.debug(f"🎤 Partial: {partial['partial']}")
            
            # Process final result
            final_result = json.loads(rec.FinalResult())
            stream.close()
            
            if final_result.get('text'):
                return self.process_recognition_result(final_result)
            
            logger.info("⏰ Listening timeout - no speech detected")
            return None
            
        except Exception as e:
            logger.error(f"❌ Single listen error: {e}")
            return None
    
    def listen(self) -> Optional[Dict]:
        """Main listen method - returns immediately if continuous listening is active"""
        if self.is_listening:
            # Return None for continuous mode - results are processed in background
            return None
        else:
            # Single listen mode
            return self.listen_once()
    
    def save_audio_sample(self, audio_data: bytes, filename: str):
        """Save audio sample for debugging"""
        try:
            samples_dir = Path("audio_samples")
            samples_dir.mkdir(exist_ok=True)
            
            filepath = samples_dir / f"{filename}.wav"
            
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            logger.debug(f"💾 Audio sample saved: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save audio sample: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        success_rate = (self.successful_recognitions / max(self.recognition_count, 1)) * 100
        
        return {
            'initialized': self.is_initialized,
            'listening': self.is_listening,
            'current_language': self.current_language,
            'available_languages': list(self.models.keys()),
            'recognition_count': self.recognition_count,
            'successful_recognitions': self.successful_recognitions,
            'success_rate': f"{success_rate:.1f}%",
            'average_confidence': f"{self.average_confidence:.2f}",
            'audio_config': {
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'chunk_size': self.chunk_size
            }
        }
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language models"""
        return list(self.models.keys())
    
    def reset_statistics(self):
        """Reset recognition statistics"""
        self.recognition_count = 0
        self.successful_recognitions = 0
        self.average_confidence = 0.0
        logger.info("📊 Recognition statistics reset")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up speech recognizer...")
        
        # Stop listening
        self.stop_listening()
        
        # Close audio system
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
            self.audio = None
        
        # Clear models
        self.models.clear()
        self.current_model = None
        
        logger.info("✅ Speech recognizer cleanup complete")


# Helper functions for model management
def download_models():
    """Helper function to download Vosk models"""
    import urllib.request
    import zipfile
    
    models_to_download = {
        'en': {
            'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
            'filename': 'vosk-model-en-us-0.22.zip'
        },
        'small': {
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
            'filename': 'vosk-model-small-en-us-0.15.zip'
        }
    }
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for lang, info in models_to_download.items():
        model_path = models_dir / info['filename']
        
        if not model_path.exists():
            print(f"📥 Downloading {lang} model...")
            try:
                urllib.request.urlretrieve(info['url'], str(model_path))
                
                # Extract
                with zipfile.ZipFile(str(model_path), 'r') as zip_ref:
                    zip_ref.extractall(str(models_dir))
                
                print(f"✅ {lang} model downloaded and extracted")
                
            except Exception as e:
                print(f"❌ Failed to download {lang} model: {e}")


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vosk Speech Recognition Test')
    parser.add_argument('--download', action='store_true', help='Download required models')
    parser.add_argument('--test', action='store_true', help='Test speech recognition')
    parser.add_argument('--language', default='en', help='Language to use (en/hi/gu)')
    parser.add_argument('--continuous', action='store_true', help='Continuous recognition mode')
    
    args = parser.parse_args()
    
    if args.download:
        download_models()
        sys.exit(0)
    
    if args.test:
        try:
            recognizer = VoskSpeechRecognizer()
            
            print("🧪 Testing Vosk Speech Recognition...")
            print("=" * 50)
            
            # Show status
            status = recognizer.get_status()
            print(f"📊 Status: {json.dumps(status, indent=2)}")
            
            # Switch language if requested
            if args.language != 'en':
                success = recognizer.switch_language(args.language)
                if not success:
                    print(f"⚠️ Language {args.language} not available")
            
            if args.continuous:
                # Continuous mode
                print("🎤 Starting continuous recognition...")
                print("   Press Ctrl+C to stop")
                
                recognizer.start_listening()
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n👋 Stopping...")
                    recognizer.stop_listening()
            else:
                # Single recognition
                print("🎤 Say something (5 second timeout)...")
                result = recognizer.listen_once(timeout=5)
                
                if result:
                    print(f"✅ Recognized: '{result['text']}'")
                    print(f"   Language: {result['language']}")
                    print(f"   Confidence: {result['confidence']:.2f}")
                else:
                    print("❌ No speech recognized")
            
            recognizer.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            sys.exit(1)
    else:
        print("Use --test to test speech recognition or --download to download models")