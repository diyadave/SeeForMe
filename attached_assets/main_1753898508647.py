#!/usr/bin/env python3
"""
Offline Smart Accessibility App for Blind Users
Complete voice assistant with emotion detection, scene description, and natural conversation
"""

import os
import sys
import logging
import time
import threading
import queue
import json
from pathlib import Path
from typing import Dict, Any, Optional
from app.camera_switcher import CameraManager
from app.gemma_connect import GemmaConnector
from app.scene_detector import SceneDetector
from app.tts_handler import TTSHandler
from app.speech_handler import VoskSpeechRecognizer
from app.name_extractor import NameExtractor

# Flask and SocketIO
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

# Configure logging with Unicode support for Windows
def setup_logging():
    """Setup logging with Unicode support for Windows emoji output"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File handler (UTF-8)
    file_handler = logging.FileHandler(
        logs_dir / 'app.log', 
        encoding='utf-8', 
        mode='a'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler with UTF-8 support for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Set up encoding for Windows console
    if sys.platform.startswith('win'):
        try:
            # Try to set console to UTF-8
            os.system('chcp 65001 >nul 2>&1')
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            # Fallback: remove emoji from logging if Unicode fails
            pass
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = setup_logging()

# Create Flask app with explicit template folder
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
socketio = SocketIO(app)

class SmartAccessibilityApp:
    """Main application class for the Smart Accessibility App"""
    
    def __init__(self):
        self.app = app
        self.socketio = socketio
        
        # Application state
        self.is_listening = False
        self.current_user = None
        self.conversation_history = []
        self.current_language = 'en'
        self.last_activity = time.time()
        
        # Initialize components (set to None for now)
        self.camera_manager = None
        self.speech_recognizer = None
        self.name_extractor = None
        self.gemma_connector = None
        self.tts_handler = None
        self.scene_detector = None
        
        # Message queues for thread communication
        self.speech_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Setup routes and socket handlers
        self.setup_routes()
        self.setup_socket_handlers()
        
        logger.info(" Smart Accessibility App initialized")
    
    def initialize_components(self):
        """Initialize all app components (real implementations)"""
        logger.info(" Initializing app components...")

        try:
            logger.info(" Initializing Camera Manager...")
            try:
                logger.info(" Initializing Camera Manager...")
                self.camera_manager = CameraManager()
            except Exception as e:
                logger.warning(f" Camera Manager initialization failed: {e}")
                self.camera_manager = None

            logger.info(" Initializing Speech Recognizer...")
            from app.speech_handler import VoskSpeechRecognizer
            self.speech_recognizer = VoskSpeechRecognizer()

            logger.info(" Initializing Name Extractor...")
            from app.name_extractor import NameExtractor
            self.name_extractor = NameExtractor()

            logger.info(" Initializing TTS Handler...")
            from app.tts_handler import TTSHandler
            self.tts_handler = TTSHandler()

            logger.info(" Initializing Scene Detector...")
            from app.scene_detector import SceneDetector
            self.scene_detector = SceneDetector()

            logger.info(" Initializing Gemma Connector...")
            from app.gemma_connect import GemmaConnector
            self.gemma_connector = GemmaConnector()

            logger.info(" All components initialized successfully ✅")
            return True

        except Exception as e:
            logger.error(f" Component initialization failed: {e}")
            return False

    
    def setup_routes(self):
        """Setup Flask routes"""
    
        @self.app.route('/')
        def index():
            """Main page"""
            try:
                return render_template('index.html')
            except Exception as e:
                logger.error(f"Failed to render template: {e}")
                return "Welcome to Smart Accessibility App (template not found)", 200
            
        @self.app.route('/status')
        def status():
            """Get app status"""
            return jsonify({
                'status': 'running',
                'components': {
                    'camera': self.camera_manager.get_camera_info() if self.camera_manager else {'status': 'not_initialized'},
                    'speech': self.speech_recognizer.get_status() if self.speech_recognizer else {'status': 'not_initialized'},
                    'gemma': self.gemma_connector.get_status() if self.gemma_connector else {'status': 'not_initialized'},
                    'tts': self.tts_handler.get_status() if self.tts_handler else {'status': 'not_initialized'}
                },
                'user': {
                    'name': self.current_user,
                    'language': self.current_language,
                    'is_listening': self.is_listening
                }
            })
    
    def setup_socket_handlers(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(" Client connected")
            emit('status_update', {'status': 'Connected to Smart Accessibility App'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(" Client disconnected")
            if self.is_listening:
                self.stop_listening()
        
        @self.socketio.on('start_listening')
        def handle_start_listening():
            logger.info(" Starting voice assistant...")
            self.start_listening()
        
        @self.socketio.on('stop_listening')
        def handle_stop_listening():
            logger.info(" Stopping voice assistant...")
            self.stop_listening()
        
        @self.socketio.on('switch_camera')
        def handle_switch_camera(data):
            camera_type = data.get('camera_type', 'front')
            logger.info(f" Switching to {camera_type} camera...")
            
            if self.camera_manager:
                success = self.camera_manager.switch_camera(camera_type)
                emit('status_update', {
                    'status': f"Camera switched to {camera_type}" if success else f"Failed to switch to {camera_type} camera"
                })
    
    def start_listening(self):
        """Start the voice assistant"""
        if self.is_listening:
            logger.info(" Already listening")
            return
    
        # Initialize components if not done
        
        self.is_listening = True
        self.socketio.emit('status_update', {'status': 'Listening...', 'active': True})
        
        # Start camera streaming
        if self.camera_manager:
            self.camera_manager.start_streaming()
        
        # Start speech recognition in background thread
        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()
        
        # Start response processing thread
        self.response_thread = threading.Thread(target=self.response_worker, daemon=True)
        self.response_thread.start()
    
    
        logger.info(" Voice assistant started")
    
    def stop_listening(self):
        """Stop the voice assistant"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        self.socketio.emit('status_update', {'status': 'Stopped', 'active': False})
        
        # Stop camera streaming
        if self.camera_manager:
            self.camera_manager.stop_streaming()
        
        # Stop speech recognition
        if self.speech_recognizer:
            self.speech_recognizer.stop_listening()
        
        logger.info(" Voice assistant stopped")
    
    def speech_worker(self):
        """Background worker for speech recognition"""
        logger.info(" Speech worker started")
        
        while self.is_listening:
            try:
                if self.speech_recognizer:
                    # Listen for speech
                    result = self.speech_recognizer.listen()
                    
                    if result and result.get('text'):
                        text = result['text']
                        confidence = result.get('confidence', 0.8)
                        language = result.get('language', 'en')
                        
                        logger.info(f" Speech recognized: '{text}' (confidence: {confidence:.2f})")
                        
                        # Update current language
                        self.current_language = language
                        
                        # Emit speech recognition result
                        self.socketio.emit('speech_recognized', {
                            'text': text,
                            'confidence': confidence,
                            'language': language
                        })
                        
                        # Add to processing queue
                        self.speech_queue.put({
                            'text': text,
                            'confidence': confidence,
                            'language': language,
                            'timestamp': time.time()
                        })
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f" Speech worker error: {e}")
                time.sleep(1)  # Longer delay on error
        
        logger.info(" Speech worker stopped")
    
    def response_worker(self):
        """Background worker for processing speech and generating responses"""
        logger.info(" Response worker started")
        
        while self.is_listening:
            try:
                # Get speech from queue (with timeout)
                try:
                    speech_data = self.speech_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                text = speech_data['text']
                language = speech_data['language']
                
                logger.info(f" Processing: '{text}'")
                self.socketio.emit('status_update', {'status': 'Thinking...'})
                
                # Process the speech input
                response = self.process_speech_input(text, language)
                
                if response:
                    logger.info(f" Generated response: '{response}'")
                    
                    # Emit response
                    self.socketio.emit('response_generated', {'text': response})
                    
                    # Speak the response
                    self.socketio.emit('status_update', {'status': 'Speaking...'})
                    self.tts_handler.speak(response, language)
                    
                    self.socketio.emit('status_update', {'status': 'Listening...', 'active': True})
                
            except Exception as e:
                logger.error(f" Response worker error: {e}")
                self.socketio.emit('error', {'message': f"Processing error: {str(e)}"})
        
        logger.info(" Response worker stopped")
    
    def process_speech_input(self, text: str, language: str) -> str:
        """Process speech input and generate appropriate response"""
        text_lower = text.lower()
        
        # Check for name extraction
        extracted_name = self.name_extractor.extract_name(text, language)
        if extracted_name:
            self.current_user = extracted_name
            logger.info(f" User name extracted: {extracted_name}")
            return self.name_extractor.get_greeting_response(extracted_name, language)
        
        # Check for scene description requests
        scene_keywords = [
            'what is there', 'tell me the scene', 'describe what you see',
            'what do you see', 'what is in front', 'what is around',
            'scene description', 'look around', 'what is happening',
            'kya hai yahan', 'scene batao', 'kya dikh raha hai',  # Hindi
            'shu che tyaan', 'scene kaho', 'shu dikhay che'  # Gujarati
        ]
        
        if any(keyword in text_lower for keyword in scene_keywords):
            return self.handle_scene_description(text, language)
        
        # Check for emotion-related queries (switch to front camera)
        emotion_keywords = [
            'i am feeling', 'i feel', 'my mood', 'how do i look',
            'i am happy', 'i am sad', 'i am angry', 'i am tired',
            'main feel kar raha hun', 'mujhe lag raha hai',  # Hindi
            'hu feel karu chu', 'mane lage che'  # Gujarati
        ]
        
        if any(keyword in text_lower for keyword in emotion_keywords):
            return self.handle_emotion_query(text, language)
        
        # General conversation - use Gemma
        return self.generate_conversational_response(text, language)
    
    def handle_scene_description(self, text: str, language: str) -> str:
        """Handle scene description requests"""
        logger.info(" Handling scene description request")
        
        # Switch to back camera for scene analysis
        if self.camera_manager:
            success = self.camera_manager.switch_camera('back')
            if not success:
                return self.get_fallback_response("Sorry, I cannot access the back camera right now.", language)
            
            # Give camera time to adjust
            time.sleep(1)
        
        # Capture frame for analysis
        frame = self.camera_manager.capture_frame() if self.camera_manager else None
        
        if frame is None:
            return self.get_fallback_response("Sorry, I cannot see anything right now.", language)
        
        try:
            # Analyze the scene
            scene_result = self.scene_detector.analyze_scene(frame)
            
            if scene_result:
                # Build comprehensive description
                description_parts = []
                
                # Scene classification
                if scene_result.get('scene'):
                    scene_name = scene_result['scene']['name']
                    scene_confidence = scene_result['scene']['confidence']
                    if scene_confidence > 0.3:
                        description_parts.append(f"I can see a {scene_name}")
                
                # Objects detected
                if scene_result.get('objects'):
                    objects = scene_result['objects'][:5]  # Top 5 objects
                    object_names = [obj['name'] for obj in objects if obj['confidence'] > 0.5]
                    if object_names:
                        if len(object_names) == 1:
                            description_parts.append(f"There is a {object_names[0]}")
                        else:
                            description_parts.append(f"I can see {', '.join(object_names[:-1])} and {object_names[-1]}")
                
                # People detected
                if scene_result.get('people'):
                    people_count = len(scene_result['people'])
                    if people_count == 1:
                        person = scene_result['people'][0]
                        emotion = person.get('emotion', 'neutral')
                        gender = person.get('gender', 'person')
                        description_parts.append(f"There is one {gender} who appears to be {emotion}")
                    elif people_count > 1:
                        description_parts.append(f"There are {people_count} people in the scene")
                
                if description_parts:
                    final_description = ". ".join(description_parts) + "."
                    
                    # Add to conversation history
                    self.add_to_conversation_history('user', text)
                    self.add_to_conversation_history('assistant', final_description)
                    
                    return final_description
            
            return self.get_fallback_response("I can see the scene but I'm having trouble describing it right now.", language)
            
        except Exception as e:
            logger.error(f" Scene analysis error: {e}")
            return self.get_fallback_response("Sorry, I'm having trouble analyzing the scene.", language)
    
    def handle_emotion_query(self, text: str, language: str) -> str:
        """Handle emotion-related queries"""
        logger.info(" Handling emotion query")
        
        # Switch to front camera for emotion detection
        if self.camera_manager:
            success = self.camera_manager.switch_camera('front')
            if not success:
                return self.get_fallback_response("Sorry, I cannot access the front camera right now.", language)
            
            # Give camera time to adjust
            time.sleep(1)
        
        # Capture frame for emotion analysis
        frame = self.camera_manager.capture_frame() if self.camera_manager else None
        
        if frame is None:
            return self.get_fallback_response("Sorry, I cannot see you right now.", language)
        
        try:
            # Analyze emotion
            scene_result = self.scene_detector.analyze_scene(frame)
            
            if scene_result and scene_result.get('people'):
                person = scene_result['people'][0]  # First person detected
                emotion = person.get('emotion', 'neutral')
                confidence = person.get('emotion_confidence', 0.5)
                
                if confidence > 0.4:
                    # Emit emotion detection result
                    self.socketio.emit('emotion_detected', {'emotion': emotion})
                    
                    # Generate empathetic response
                    emotion_responses = {
                        'happy': "I can see you're feeling happy! That's wonderful. Your positive energy is contagious.",
                        'sad': "I notice you seem a bit sad. I'm here for you. Would you like to talk about what's on your mind?",
                        'angry': "You appear to be feeling frustrated. Take a deep breath. I'm here to listen if you need to express your feelings.",
                        'fear': "I can sense you might be feeling anxious. Remember that you're safe, and I'm here to support you.",
                        'surprise': "You look surprised! Something interesting must have happened.",
                        'disgust': "I can see you're not comfortable with something right now.",
                        'neutral': "You look calm and composed. How are you feeling today?"
                    }
                    
                    response = emotion_responses.get(emotion, f"I can see you're feeling {emotion}.")
                    
                    # Add to conversation history
                    self.add_to_conversation_history('user', text)
                    self.add_to_conversation_history('assistant', response)
                    
                    return response
            
            return self.get_fallback_response("I can see you but I'm having trouble reading your emotions right now.", language)
            
        except Exception as e:
            logger.error(f" Emotion analysis error: {e}")
            return self.get_fallback_response("Sorry, I'm having trouble analyzing your emotions.", language)
    
    def generate_conversational_response(self, text: str, language: str) -> str:
        """Generate conversational response using Gemma"""
        logger.info(" Generating conversational response")
        
        # Build context for Gemma
        context = {
            'user_name': self.current_user,
            'current_emotion': 'Neutral',
            'language': language,
            'conversation_history': self.conversation_history[-5:]  # Last 5 exchanges
        }
        
        # Try to get response from Gemma
        if self.gemma_connector and self.gemma_connector.is_available():
            try:
                response = self.gemma_connector.generate_response(text, context, timeout=10)
                if response:
                    # Add to conversation history
                    self.add_to_conversation_history('user', text)
                    self.add_to_conversation_history('assistant', response)
                    
                    return response
            except Exception as e:
                logger.warning(f" Gemma response error: {e}")
        
        # Fallback to default responses
        return self.get_fallback_response(text, language)
    
    def get_fallback_response(self, text: str, language: str) -> str:
        """Generate fallback response when Gemma is not available"""
        text_lower = text.lower()
        
        # Language-specific fallback responses
        fallback_responses = {
            'en': {
                'greeting': ["Hello! How can I help you today?", "Hi there! I'm here to assist you.", "Welcome! What can I do for you?"],
                'thanks': ["You're welcome!", "Happy to help!", "Glad I could assist you!"],
                'how_are_you': ["I'm doing well, thank you for asking! How are you?", "I'm here and ready to help! How are you feeling?"],
                'default': ["I understand. Can you tell me more?", "That's interesting. What else would you like to know?", "I'm here to help you with whatever you need."]
            },
            'hi': {
                'greeting': ["Namaste! Main aapki kaise madad kar sakta hun?", "Namaskar! Kya chahiye aapko?"],
                'thanks': ["Aapka swagat hai!", "Khushi mili madad karke!"],
                'how_are_you': ["Main theek hun, dhanyawad! Aap kaise hain?"],
                'default': ["Samjha. Aur batayiye?", "Accha, kya aur puchna chahte hain?"]
            },
            'gu': {
                'greeting': ["Namaste! Hu tamari kevi madad kari saku?", "Namaskar! Tame shu chacho?"],
                'thanks': ["Tamaru swagat che!", "Madad karine khushi thayi!"],
                'how_are_you': ["Hu saras chu, thanks! Tame kem cho?"],
                'default': ["Samjay gyu. Avar shu puchvo che?", "Saras, koi avar prashn?"]
            }
        }
        
        responses = fallback_responses.get(language, fallback_responses['en'])
        
        # Choose appropriate response based on input
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'namaste', 'namaskar']):
            import random
            return random.choice(responses['greeting'])
        elif any(word in text_lower for word in ['thank', 'thanks', 'dhanyawad']):
            import random
            return random.choice(responses['thanks'])
        elif any(word in text_lower for word in ['how are you', 'kaise ho', 'kem cho']):
            import random
            return random.choice(responses['how_are_you'])
        else:
            import random
            return random.choice(responses['default'])
    
    def add_to_conversation_history(self, role: str, text: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'text': text,
            'timestamp': time.time()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def cleanup(self):
        """Clean up resources"""
        logger.info(" Cleaning up app resources...")
        
        # Stop listening
        self.stop_listening()
        
        # Cleanup components
        if self.camera_manager:
            self.camera_manager.cleanup()
        
        if self.speech_recognizer:
            self.speech_recognizer.cleanup()
        
        if self.gemma_connector:
            self.gemma_connector.cleanup()
        
        if self.tts_handler:
            self.tts_handler.cleanup()
        
        if self.scene_detector:
            self.scene_detector.cleanup()
        
        logger.info(" App cleanup complete")
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the application"""
        logger.info(f" Starting Smart Accessibility App on {host}:{port}")
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        except KeyboardInterrupt:
            logger.info(" Application interrupted by user")
        except Exception as e:
            logger.error(f" Application error: {e}")
        finally:
            self.cleanup()


# Main execution
if __name__ == "__main__":
    import argparse
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Smart Accessibility App for Blind Users')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create and run app
    smart_app = SmartAccessibilityApp()
    
    logger.info(" Smart Accessibility App for Blind Users")
    logger.info("=" * 50)
    logger.info("Features:")
    logger.info("   Multi-language voice recognition (English/Hindi/Gujarati)")
    logger.info("   Text-to-speech with language support")
    logger.info("   Real-time emotion detection")
    logger.info("   Scene and object description")
    logger.info("   Natural conversation with Gemma 3n")
    logger.info("   Smart camera switching")
    logger.info("   Fully offline operation")
    logger.info("=" * 50)
    
    try:
        smart_app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f" Failed to start application: {e}")
        sys.exit(1)