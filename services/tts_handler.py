#!/usr/bin/env python3
"""
Multi-language Text-to-Speech Handler
Supports offline English TTS with pyttsx3 and online Hindi/Gujarati with gTTS fallback
"""

import os
import time
import logging
import threading
import tempfile
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from io import BytesIO
import hashlib
import subprocess
import platform

# TTS imports with fallbacks
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False
    logging.warning("⚠️ pyttsx3 not installed. English TTS limited. Run: pip install pyttsx3")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    gTTS = None
    GTTS_AVAILABLE = False
    logging.warning("⚠️ gTTS not installed. Hindi/Gujarati TTS limited. Run: pip install gtts")

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    logging.warning("⚠️ pygame not installed. Using system audio player. Run: pip install pygame")

logger = logging.getLogger(__name__)

class TTSHandler:
    """Advanced Text-to-Speech handler with multi-language support"""
    
    def __init__(self):
        logger.info("🔊 Initializing TTS Handler...")
        
        # TTS engines
        self.pyttsx3_engine = None
        self.gtts_available = False
        self.pygame_available = PYGAME_AVAILABLE
        
        # Configuration
        self.default_language = 'en'
        self.speech_rate = 150  # Words per minute
        self.volume = 0.9
        
        # Language settings
        self.language_config = {
            'en': {
                'engine': 'pyttsx3',
                'voice_id': None,  # Will be set during initialization
                'rate': 150,
                'volume': 0.9
            },
            'hi': {
                'engine': 'gtts',
                'gtts_lang': 'hi',
                'rate': 130,
                'volume': 0.9,
                'fallback_text': 'I understand Hindi but cannot speak it right now.'
            },
            'gu': {
                'engine': 'gtts',
                'gtts_lang': 'gu',
                'rate': 130,
                'volume': 0.9,
                'fallback_text': 'I understand Gujarati but cannot speak it right now.'
            }
        }
        
        # State management
        self.is_speaking = False
        self.speaking_thread = None
        self.speech_queue = queue.Queue()
        self.stop_flag = threading.Event()
        
        # Performance tracking
        self.speech_count = 0
        self.successful_speeches = 0
        self.average_speech_time = 0.0
        
        # Cache for TTS audio files
        self.audio_cache = {}
        self.cache_dir = Path("tts_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize engines
        self.initialize_engines()
        
        logger.info("✅ TTS Handler initialized")
    
    def initialize_engines(self):
        """Initialize TTS engines"""
        
        # Initialize pyttsx3 for English
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                
                # Configure pyttsx3
                self.configure_pyttsx3()
                
                logger.info("✅ pyttsx3 engine initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize pyttsx3: {e}")
                self.pyttsx3_engine = None
        
        # Check gTTS availability
        if GTTS_AVAILABLE:
            try:
                # Test internet connection for gTTS
                test_response = requests.get('https://translate.google.com', timeout=3)
                if test_response.status_code == 200:
                    self.gtts_available = True
                    logger.info("✅ gTTS available (internet connection active)")
                else:
                    self.gtts_available = False
                    logger.warning("⚠️ gTTS unavailable (no internet connection)")
            except:
                self.gtts_available = False
                logger.warning("⚠️ gTTS unavailable (no internet connection)")
        
        # Initialize pygame mixer
        if PYGAME_AVAILABLE and not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
                logger.info("✅ pygame audio mixer initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize pygame mixer: {e}")
                self.pygame_available = False
    
    def configure_pyttsx3(self):
        """Configure pyttsx3 engine settings"""
        if not self.pyttsx3_engine:
            return
        
        try:
            # Get available voices
            voices = self.pyttsx3_engine.getProperty('voices')
            
            if voices:
                # Try to find a good English voice
                best_voice = None
                
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = voice.id.lower()
                    
                    # Prefer female voices for accessibility
                    if any(keyword in voice_name for keyword in ['zira', 'hazel', 'female', 'woman']):
                        best_voice = voice.id
                        break
                    elif any(keyword in voice_id for keyword in ['english', 'en-us', 'en_us']):
                        best_voice = voice.id
                
                if best_voice:
                    self.pyttsx3_engine.setProperty('voice', best_voice)
                    self.language_config['en']['voice_id'] = best_voice
                    logger.info(f"🎭 Selected voice: {best_voice}")
                else:
                    # Use first available voice
                    self.pyttsx3_engine.setProperty('voice', voices[0].id)
                    logger.info(f"🎭 Using default voice: {voices[0].name}")
            
            # Set speech rate
            self.pyttsx3_engine.setProperty('rate', self.speech_rate)
            
            # Set volume
            self.pyttsx3_engine.setProperty('volume', self.volume)
            
            logger.info(f"🔧 pyttsx3 configured: rate={self.speech_rate}, volume={self.volume}")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure pyttsx3: {e}")
    
    def speak(self, text: str, language: str = 'en', priority: bool = False):
        """
        Speak text in specified language
        
        Args:
            text: Text to speak
            language: Language code ('en', 'hi', 'gu')
            priority: If True, speak immediately (interrupt current speech)
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for TTS")
            return
        
        text = text.strip()
        logger.info(f"🔊 Speaking [{language}]: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Add to speech queue
        speech_item = {
            'text': text,
            'language': language,
            'timestamp': time.time(),
            'priority': priority
        }
        
        if priority:
            # Stop current speech and clear queue
            self.stop_speaking()
            # Add to front of queue
            temp_queue = queue.Queue()
            temp_queue.put(speech_item)
            while not self.speech_queue.empty():
                try:
                    temp_queue.put(self.speech_queue.get_nowait())
                except queue.Empty:
                    break
            self.speech_queue = temp_queue
        else:
            self.speech_queue.put(speech_item)
        
        # Start speech worker if not running
        if not self.is_speaking:
            self.start_speech_worker()
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.is_speaking:
            self.stop_flag.set()
            
            # Stop pygame if playing
            if self.pygame_available:
                try:
                    pygame.mixer.music.stop()
                except:
                    pass
            
            # Stop pyttsx3 if speaking
            if self.pyttsx3_engine:
                try:
                    self.pyttsx3_engine.stop()
                except:
                    pass
            
            logger.info("⏹️ Speech stopped")
    
    def start_speech_worker(self):
        """Start background speech worker"""
        if self.is_speaking:
            return
        
        self.is_speaking = True
        self.stop_flag.clear()
        self.speaking_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speaking_thread.start()
    
    def speech_worker(self):
        """Background worker for speech processing"""
        logger.info("🎭 Speech worker started")
        
        while self.is_speaking and not self.stop_flag.is_set():
            try:
                # Get speech item from queue
                try:
                    speech_item = self.speech_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                if self.stop_flag.is_set():
                    break
                
                # Process speech
                start_time = time.time()
                success = self.process_speech_item(speech_item)
                speech_time = time.time() - start_time
                
                # Update statistics
                self.speech_count += 1
                if success:
                    self.successful_speeches += 1
                    self.update_average_speech_time(speech_time)
                
            except Exception as e:
                logger.error(f"❌ Speech worker error: {e}")
        
        self.is_speaking = False
        logger.info("🎭 Speech worker stopped")
    
    def process_speech_item(self, speech_item: Dict[str, Any]) -> bool:
        """Process individual speech item"""
        if self.stop_flag.is_set():
            return False
            
        text = speech_item['text']
        language = speech_item['language']
        
        try:
            if language == 'en':
                return self.speak_english(text)
            elif language in ['hi', 'gu']:
                return self.speak_gtts(text, language)
            else:
                logger.warning(f"⚠️ Unsupported language: {language}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to process speech item: {e}")
            return False
    
    def speak_english(self, text: str) -> bool:
        """Speak English text using pyttsx3"""
        if self.stop_flag.is_set():
            return False
            
        if not self.pyttsx3_engine:
            logger.error("❌ pyttsx3 engine not available")
            return False
        
        try:
            # Use pyttsx3 for offline English TTS
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            
            if not self.stop_flag.is_set():
                logger.info("✅ English speech completed")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 speech failed: {e}")
            return False
    
    def speak_gtts(self, text: str, language: str) -> bool:
        """Speak text using gTTS (Google Text-to-Speech)"""
        if self.stop_flag.is_set():
            return False
            
        if not self.gtts_available:
            logger.warning(f"⚠️ gTTS not available for {language}")
            # Fallback to English explanation
            fallback_text = self.language_config[language]['fallback_text']
            return self.speak_english(fallback_text)
        
        try:
            # Check cache first
            cache_key = self.get_cache_key(text, language)
            cached_file = self.get_cached_audio(cache_key)
            
            if cached_file and cached_file.exists():
                logger.info(f"🗄️ Using cached audio for {language}")
                return self.play_audio_file(cached_file)
            
            # Generate TTS using gTTS
            gtts_lang = self.language_config[language]['gtts_lang']
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to cache
            cache_file = self.cache_dir / f"{cache_key}.mp3"
            tts.save(str(cache_file))
            
            if self.stop_flag.is_set():
                return False
            
            # Play audio
            success = self.play_audio_file(cache_file)
            
            if success:
                logger.info(f"✅ {language.upper()} speech completed")
            else:
                logger.error(f"❌ Failed to play {language} audio")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ gTTS speech failed for {language}: {e}")
            # Fallback to English explanation
            fallback_text = self.language_config[language]['fallback_text']
            return self.speak_english(fallback_text)
    
    def play_audio_file(self, file_path: Path) -> bool:
        """Play audio file using pygame or system player"""
        if self.stop_flag.is_set():
            return False
            
        try:
            if self.pygame_available:
                # Use pygame for audio playback
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy() and not self.stop_flag.is_set():
                    time.sleep(0.1)
                
                return not self.stop_flag.is_set()
            else:
                # Fallback to system audio player
                return self.play_with_system_player(file_path)
                
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            return False
    
    def play_with_system_player(self, file_path: Path) -> bool:
        """Play audio using system audio player"""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Use Windows Media Player
                subprocess.run(["start", str(file_path)], shell=True, check=True)
            elif system == "darwin":
                # Use macOS afplay
                subprocess.run(["afplay", str(file_path)], check=True)
            else:
                # Use Linux aplay or mpg123
                try:
                    subprocess.run(["mpg123", str(file_path)], check=True)
                except FileNotFoundError:
                    try:
                        subprocess.run(["aplay", str(file_path)], check=True)
                    except FileNotFoundError:
                        logger.error("❌ No suitable audio player found")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System audio player failed: {e}")
            return False
    
    def get_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for text and language"""
        cache_string = f"{text}_{language}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_audio(self, cache_key: str) -> Optional[Path]:
        """Get cached audio file if it exists"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        return cache_file if cache_file.exists() else None
    
    def update_average_speech_time(self, speech_time: float):
        """Update average speech time"""
        if self.average_speech_time == 0:
            self.average_speech_time = speech_time
        else:
            self.average_speech_time = (
                0.2 * speech_time + 0.8 * self.average_speech_time
            )
    
    def clear_cache(self):
        """Clear TTS audio cache"""
        try:
            for cache_file in self.cache_dir.glob("*.mp3"):
                cache_file.unlink()
            logger.info("🗑️ TTS cache cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_speaking()
            
            # Stop speech worker
            if self.speaking_thread and self.speaking_thread.is_alive():
                self.speaking_thread.join(timeout=2)
            
            # Cleanup pyttsx3
            if self.pyttsx3_engine:
                try:
                    self.pyttsx3_engine.stop()
                except:
                    pass
            
            # Cleanup pygame
            if self.pygame_available:
                try:
                    pygame.mixer.quit()
                except:
                    pass
            
            # Clear queues
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("🧹 TTS handler cleaned up")
            
        except Exception as e:
            logger.error(f"❌ TTS cleanup error: {e}")
    
    def get_status(self) -> Dict:
        """Get current status"""
        success_rate = (
            (self.successful_speeches / self.speech_count * 100) 
            if self.speech_count > 0 else 0
        )
        
        return {
            'is_speaking': self.is_speaking,
            'pyttsx3_available': self.pyttsx3_engine is not None,
            'gtts_available': self.gtts_available,
            'pygame_available': self.pygame_available,
            'speech_count': self.speech_count,
            'success_rate': f"{success_rate:.1f}%",
            'average_speech_time': f"{self.average_speech_time:.2f}s",
            'queue_size': self.speech_queue.qsize() if hasattr(self.speech_queue, 'qsize') else 0,
            'cache_files': len(list(self.cache_dir.glob("*.mp3"))) if self.cache_dir.exists() else 0
        }

# Test function
if __name__ == "__main__":
    tts = TTSHandler()
    
    # Test English
    tts.speak("Hello, this is a test of the English text to speech system.")
    time.sleep(3)
    
    # Test Hindi (if available)
    tts.speak("नमस्ते, यह हिंदी टेक्स्ट टू स्पीच का परीक्षण है।", "hi")
    time.sleep(3)
    
    # Test status
    print(f"TTS Status: {tts.get_status()}")
    
    # Cleanup
    tts.cleanup()
