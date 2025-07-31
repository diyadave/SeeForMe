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

try:
    import pyttsx3  # pip install pyttsx3
except ImportError:
    pyttsx3 = None
    logging.error("❌ pyttsx3 not installed. Run: pip install pyttsx3")

try:
    from gtts import gTTS  # pip install gtts
except ImportError:
    gTTS = None
    logging.warning("⚠️ gTTS not installed. Hindi/Gujarati TTS will be limited. Run: pip install gtts")

try:
    import pygame  # pip install pygame
    pygame.mixer.init()
except ImportError:
    pygame = None
    logging.warning("⚠️ pygame not installed. Using system audio player. Run: pip install pygame")

logger = logging.getLogger(__name__)


class TTSHandler:
    """Advanced Text-to-Speech handler with multi-language support"""
    
    def __init__(self):
        logger.info("🔊 Initializing TTS Handler...")
        
        # TTS engines
        self.pyttsx3_engine = None
        self.gtts_available = False
        self.pygame_available = bool(pygame)
        
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
        if pyttsx3:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                
                # Configure pyttsx3
                self.configure_pyttsx3()
                
                logger.info("✅ pyttsx3 engine initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize pyttsx3: {e}")
                self.pyttsx3_engine = None
        
        # Check gTTS availability
        if gTTS:
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
        if pygame and not pygame.mixer.get_init():
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
    
    def start_speech_worker(self):
        """Start background speech worker"""
        if self.is_speaking:
            return
        
        self.is_speaking = True
        self.speaking_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speaking_thread.start()
    
    def speech_worker(self):
        """Background worker for speech processing"""
        logger.info("🎭 Speech worker started")
        
        while self.is_speaking:
            try:
                # Get speech item from queue
                try:
                    speech_item = self.speech_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
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
        
        logger.info("🎭 Speech worker stopped")
    
    def process_speech_item(self, speech_item: Dict[str, Any]) -> bool:
        """Process individual speech item"""
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
        if not self.pyttsx3_engine:
            logger.error("❌ pyttsx3 engine not available")
            return False
        
        try:
            # Use pyttsx3 for offline English TTS
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            logger.info("✅ English speech completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 speech failed: {e}")
            return False
    
    def speak_gtts(self, text: str, language: str) -> bool:
        """Speak text using gTTS (Google Text-to-Speech)"""
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
        try:
            if self.pygame_available:
                # Use pygame for audio playback
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                return True
            else:
                # Fallback to system audio player
                if os.name == 'nt':  # Windows
                    os.system(f'start "" "{file_path}"')
                elif os.name == 'posix':  # macOS/Linux
                    if os.system('which afplay > /dev/null 2>&1') == 0:  # macOS
                        os.system(f'afplay "{file_path}"')
                    elif os.system('which mpg123 > /dev/null 2>&1') == 0:  # Linux
                        os.system(f'mpg123 "{file_path}"')
                    else:
                        os.system(f'play "{file_path}"')  # SoX
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            return False
    
    def get_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for text and language"""
        content = f"{text}_{language}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_audio(self, cache_key: str) -> Optional[Path]:
        """Get cached audio file if exists"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        return cache_file if cache_file.exists() else None
    
    def update_average_speech_time(self, speech_time: float):
        """Update average speech time statistics"""
        if self.successful_speeches == 1:
            self.average_speech_time = speech_time
        else:
            # Running average
            self.average_speech_time = (
                (self.average_speech_time * (self.successful_speeches - 1) + speech_time) 
                / self.successful_speeches
            )
    
    def stop_speaking(self):
        """Stop current speech and clear queue"""
        logger.info("🛑 Stopping speech...")
        
        # Stop pyttsx3
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except:
                pass
        
        # Stop pygame
        if self.pygame_available:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        # Clear speech queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        
        # Stop speech worker
        self.is_speaking = False
        
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.speaking_thread.join(timeout=2)
    
    def is_currently_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.is_speaking and not self.speech_queue.empty()
    
    def get_queue_size(self) -> int:
        """Get current speech queue size"""
        return self.speech_queue.qsize()
    
    def set_speech_rate(self, rate: int):
        """Set speech rate for pyttsx3"""
        self.speech_rate = max(50, min(300, rate))  # Clamp between 50-300
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('rate', self.speech_rate)
        logger.info(f"🔧 Speech rate set to: {self.speech_rate}")
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))  # Clamp between 0-1
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('volume', self.volume)
        logger.info(f"🔊 Volume set to: {self.volume}")
    
    def get_available_voices(self) -> List[Dict[str, str]]:
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
                        'gender': 'unknown'
                    })
            except Exception as e:
                logger.error(f"❌ Failed to get voices: {e}")
        
        return voices
    
    def set_voice(self, voice_id: str) -> bool:
        """Set specific voice by ID"""
        if not self.pyttsx3_engine:
            return False
        
        try:
            voices = self.pyttsx3_engine.getProperty('voices')
            voice_ids = [voice.id for voice in voices]
            
            if voice_id in voice_ids:
                self.pyttsx3_engine.setProperty('voice', voice_id)
                self.language_config['en']['voice_id'] = voice_id
                logger.info(f"🎭 Voice changed to: {voice_id}")
                return True
            else:
                logger.warning(f"⚠️ Voice ID not found: {voice_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to set voice: {e}")
            return False
    
    def clear_cache(self):
        """Clear TTS audio cache"""
        try:
            for cache_file in self.cache_dir.glob("*.mp3"):
                cache_file.unlink()
            self.audio_cache.clear()
            logger.info("🗑️ TTS cache cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get TTS usage statistics"""
        success_rate = (self.successful_speeches / self.speech_count * 100) if self.speech_count > 0 else 0
        
        return {
            'total_speeches': self.speech_count,
            'successful_speeches': self.successful_speeches,
            'success_rate': round(success_rate, 2),
            'average_speech_time': round(self.average_speech_time, 2),
            'queue_size': self.get_queue_size(),
            'is_speaking': self.is_currently_speaking(),
            'cache_size': len(list(self.cache_dir.glob("*.mp3"))),
            'engines_available': {
                'pyttsx3': bool(self.pyttsx3_engine),
                'gtts': self.gtts_available,
                'pygame': self.pygame_available
            }
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_speaking()


# Example usage and test functions
def main():
    """Test the TTS Handler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize TTS handler
    tts = TTSHandler()
    
    # Display statistics
    print("\n📊 TTS Handler Status:")
    stats = tts.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test different languages
    print("\n🎭 Testing TTS...")
    
    # English test
    tts.speak("Hello! This is a test of the English text-to-speech system.", 'en')
    
    # Hindi test (requires internet)
    tts.speak("नमस्ते! यह हिंदी टेक्स्ट-टू-स्पीच का परीक्षण है।", 'hi')
    
    # Gujarati test (requires internet)
    tts.speak("નમસ્તે! આ ગુજરાતી ટેક્સ્ટ-ટુ-સ્પીચનું પરીક્ષણ છે.", 'gu')
    
    # Wait for completion
    time.sleep(2)
    while tts.is_currently_speaking():
        print("⏳ Speaking...")
        time.sleep(1)
    
    # Final statistics
    print("\n📊 Final Statistics:")
    final_stats = tts.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ TTS test completed!")


if __name__ == "__main__":
    main()