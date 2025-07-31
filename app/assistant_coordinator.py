#!/usr/bin/env python3
"""
Assistant Coordinator - Central orchestration for SeeForMe
Manages all components and coordinates voice, vision, and AI processing
"""

import logging
import threading
import time
import queue
from typing import Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class AssistantCoordinator:
    """Central coordinator for all SeeForMe components"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.is_active = False
        self.is_listening = False
        self.camera_permission = False
        
        # Component instances
        self.speech_handler = None
        self.camera_switcher = None
        self.emotion_detector = None
        self.scene_detector = None
        self.gemma_connector = None
        self.tts_handler = None
        self.name_extractor = None
        
        # User context
        self.user_context = {
            'name': 'friend',
            'current_emotion': 'neutral',
            'last_scene': 'unknown',
            'conversation_history': [],
            'language': 'en'
        }
        
        # Processing queue
        self.processing_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Initialize components
        self.initialize_components()
        
        # Start worker thread
        self.start_worker()
        
        logger.info("✅ Assistant Coordinator initialized")
    
    def initialize_components(self):
        """Initialize all assistant components"""
        try:
            # Initialize speech recognition
            from .speech_handler import SpeechHandler
            self.speech_handler = SpeechHandler()
            self.speech_handler.set_callback(self.on_speech_recognized)
            logger.info("✅ Speech handler initialized")
            
            # Initialize camera switching
            from .camera_switcher import CameraSwitcher
            self.camera_switcher = CameraSwitcher()
            logger.info("✅ Camera switcher initialized")
            
            # Initialize emotion detection
            from .emotion_detector import EmotionDetector
            self.emotion_detector = EmotionDetector()
            logger.info("✅ Emotion detector initialized")
            
            # Initialize scene detection
            from .scene_detector import SceneDetector
            self.scene_detector = SceneDetector()
            logger.info("✅ Scene detector initialized")
            
            # Initialize Gemma connector
            from .gemma_connect import GemmaConnector
            self.gemma_connector = GemmaConnector()
            logger.info("✅ Gemma connector initialized")
            
            # Initialize TTS
            from .tts_handler import TTSHandler
            self.tts_handler = TTSHandler()
            logger.info("✅ TTS handler initialized")
            
            # Initialize name extractor
            from .name_extractor import NameExtractor
            self.name_extractor = NameExtractor()
            logger.info("✅ Name extractor initialized")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def start_worker(self):
        """Start the processing worker thread"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("🔄 Worker thread started")
    
    def _worker_loop(self):
        """Main processing loop"""
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.processing_queue.get(timeout=1.0)
                self._process_task(task)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Worker error: {e}")
    
    def _process_task(self, task):
        """Process a single task"""
        task_type = task.get('type')
        data = task.get('data', {})
        
        if task_type == 'voice_input':
            self._handle_voice_input(data)
        elif task_type == 'camera_analysis':
            self._handle_camera_analysis(data)
        elif task_type == 'generate_response':
            self._handle_response_generation(data)
        else:
            logger.warning(f"⚠️ Unknown task type: {task_type}")
    
    def start_listening(self):
        """Start voice recognition"""
        if self.speech_handler and not self.is_listening:
            self.is_listening = True
            self.speech_handler.start_listening()
            self._emit_status_update()
            logger.info("🎤 Voice recognition started")
    
    def stop_listening(self):
        """Stop voice recognition"""
        if self.speech_handler and self.is_listening:
            self.is_listening = False
            self.speech_handler.stop_listening()
            self._emit_status_update()
            logger.info("🛑 Voice recognition stopped")
    
    def set_camera_permission(self, permitted: bool):
        """Set camera permission status"""
        self.camera_permission = permitted
        if self.camera_switcher:
            self.camera_switcher.set_permission(permitted)
        logger.info(f"📹 Camera permission: {'granted' if permitted else 'denied'}")
    
    def process_voice_input(self, text: str, language: str = 'en', confidence: float = 1.0):
        """Process voice input from user"""
        task = {
            'type': 'voice_input',
            'data': {
                'text': text,
                'language': language,
                'confidence': confidence,
                'timestamp': time.time()
            }
        }
        self.processing_queue.put(task)
    
    def on_speech_recognized(self, text: str, language: str = 'en', confidence: float = 1.0):
        """Callback for speech recognition"""
        logger.info(f"🧠 Processing speech: '{text}' (language: {language}, confidence: {confidence:.2f})")
        
        # Process voice input immediately
        self._handle_voice_input({
            'text': text,
            'language': language,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Emit to client
        self.socketio.emit('speech_recognized', {
            'text': text,
            'language': language,
            'confidence': confidence
        })
    
    def _handle_voice_input(self, data):
        """Handle voice input processing"""
        text = data['text']
        language = data['language']
        confidence = data['confidence']
        
        logger.info(f"🧠 Processing: '{text}' ({language}, {confidence:.2f})")
        
        # Update conversation history
        self.user_context['conversation_history'].append({
            'user': text,
            'timestamp': time.time()
        })
        
        # Extract name if mentioned
        if self.name_extractor:
            extracted_name = self.name_extractor.extract_name(text, language)
            if extracted_name:
                self.user_context['name'] = extracted_name
                logger.info(f"👤 User name: {extracted_name}")
        
        # Analyze intent and determine camera mode
        intent = self._analyze_intent(text)
        camera_mode = self._determine_camera_mode(intent, text)
        
        # Emit visual feedback
        self.socketio.emit('processing_intent', {
            'intent': intent,
            'camera_mode': camera_mode,
            'text': text
        })
        
        # Switch camera if needed and analyze
        if camera_mode and self.camera_permission:
            self._switch_and_analyze_camera(camera_mode, text, language)
        else:
            # Generate response without vision
            self._generate_text_response(text, language, intent)
    
    def _analyze_intent(self, text: str) -> str:
        """Analyze user intent from text"""
        text_lower = text.lower()
        
        # Scene/environment analysis intents
        scene_keywords = [
            'what do you see', 'what\'s there', 'what is there', 'where am i',
            'what\'s in front', 'what is in front', 'describe surroundings',
            'look around', 'what\'s around', 'environment', 'scene'
        ]
        
        # Emotion/self analysis intents
        emotion_keywords = [
            'how do i look', 'my mood', 'how am i feeling', 'my emotion',
            'my face', 'my expression', 'do i look', 'am i smiling'
        ]
        
        # Weather/time intents
        weather_keywords = [
            'weather', 'how\'s the weather', 'what\'s the view', 'time of day',
            'is it sunny', 'is it bright', 'lighting'
        ]
        
        # People detection intents
        people_keywords = [
            'who is there', 'is someone there', 'anyone there', 'people',
            'person in front', 'who\'s there'
        ]
        
        for keyword in scene_keywords:
            if keyword in text_lower:
                return 'scene_analysis'
        
        for keyword in emotion_keywords:
            if keyword in text_lower:
                return 'emotion_analysis'
        
        for keyword in weather_keywords:
            if keyword in text_lower:
                return 'weather_analysis'
        
        for keyword in people_keywords:
            if keyword in text_lower:
                return 'people_detection'
        
        return 'general_conversation'
    
    def _determine_camera_mode(self, intent: str, text: str) -> Optional[str]:
        """Determine which camera to use based on intent"""
        if intent in ['emotion_analysis']:
            return 'front'
        elif intent in ['scene_analysis', 'weather_analysis', 'people_detection']:
            return 'back'
        return None
    
    def _switch_and_analyze_camera(self, camera_mode: str, text: str, language: str):
        """Switch camera and perform analysis"""
        if not self.camera_switcher:
            return
        
        # Switch camera
        success = self.camera_switcher.switch_to(camera_mode)
        if not success:
            self.socketio.emit('assistant_response', {
                'text': 'I cannot access the camera right now. Please check camera permissions.',
                'emotion': 'apologetic',
                'speak': True
            })
            return
        
        # Emit camera switching status
        self.socketio.emit('camera_switched', {
            'mode': camera_mode,
            'status': 'analyzing'
        })
        
        # Capture frame
        frame = self.camera_switcher.capture_frame()
        if frame is None:
            self.socketio.emit('assistant_response', {
                'text': 'I cannot see clearly right now. Please make sure the camera is not blocked.',
                'emotion': 'apologetic',
                'speak': True
            })
            return
        
        # Perform analysis based on camera mode
        analysis_results = {}
        
        if camera_mode == 'front' and self.emotion_detector:
            # Analyze user's emotion
            emotion_data = self.emotion_detector.detect_emotion(frame)
            analysis_results['emotion'] = emotion_data
            if emotion_data:
                self.user_context['current_emotion'] = emotion_data.get('emotion', 'neutral')
        
        if camera_mode == 'back' and self.scene_detector:
            # Analyze scene, objects, and people
            scene_data = self.scene_detector.analyze_scene(frame)
            analysis_results['scene'] = scene_data
            if scene_data:
                self.user_context['last_scene'] = scene_data.get('scene_type', 'unknown')
                
                # Check for people and their emotions
                if 'people' in scene_data and scene_data['people']:
                    people_emotions = []
                    for person_bbox in scene_data['people']:
                        person_frame = self._extract_person_region(frame, person_bbox)
                        if person_frame is not None and self.emotion_detector:
                            person_emotion = self.emotion_detector.detect_emotion(person_frame)
                            people_emotions.append(person_emotion)
                    analysis_results['people_emotions'] = people_emotions
        
        # Generate contextual response
        self._generate_vision_response(text, language, analysis_results)
    
    def _generate_vision_response(self, text: str, language: str, analysis_results: Dict):
        """Generate response based on vision analysis"""
        context = {
            'user_input': text,
            'language': language,
            'user_context': self.user_context,
            'vision_results': analysis_results
        }
        
        # Always use fallback for immediate response - no dependency on complex AI models
        response = self._fallback_vision_response(analysis_results)
        
        # Emit response
        self.socketio.emit('assistant_response', {
            'text': response,
            'emotion': 'helpful',
            'speak': True,
            'vision_data': analysis_results
        })
        
        # Speak response
        if self.tts_handler:
            self.tts_handler.speak(response, language)
        
        # Update conversation history
        self.user_context['conversation_history'].append({
            'assistant': response,
            'timestamp': time.time()
        })
    
    def _generate_text_response(self, text: str, language: str, intent: str):
        """Generate text-only response"""
        logger.info(f"📝 Generating text response for: '{text}' (intent: {intent}, language: {language})")
        
        context = {
            'user_input': text,
            'language': language,
            'user_context': self.user_context,
            'intent': intent
        }
        
        # Use your Gemma 3n integration for proper AI responses
        try:
            if self.gemma_connect:
                response = self.gemma_connect.get_response(text, self.user_context)
            else:
                response = self._fallback_text_response(text, intent)
        except Exception as e:
            logger.error(f"❌ Gemma 3n error: {e}")
            response = self._fallback_text_response(text, intent)
        logger.info(f"💬 Generated response: '{response}'")
        
        # Emit response
        self.socketio.emit('assistant_response', {
            'text': response,
            'emotion': 'friendly',
            'speak': True
        })
        
        # Speak response
        if self.tts_handler:
            logger.info(f"🔊 Speaking text response: '{response[:50]}...' in {language}")
            self.tts_handler.speak(response, language)
        else:
            logger.error("❌ TTS handler not available for text response!")
        
        # Update conversation history
        self.user_context['conversation_history'].append({
            'assistant': response,
            'timestamp': time.time()
        })
    
    def _fallback_vision_response(self, analysis_results: Dict) -> str:
        """Fallback response when Gemma is unavailable"""
        responses = []
        
        if 'emotion' in analysis_results:
            emotion = analysis_results['emotion'].get('emotion', 'neutral')
            confidence = analysis_results['emotion'].get('confidence', 0.0)
            if confidence > 0.5:
                responses.append(f"You look {emotion.lower()}.")
        
        if 'scene' in analysis_results:
            scene = analysis_results['scene']
            scene_type = scene.get('scene_type', 'an indoor space')
            objects = scene.get('objects', [])
            
            responses.append(f"You are in {scene_type}.")
            if objects:
                obj_list = ', '.join(objects[:3])
                responses.append(f"I can see {obj_list}.")
        
        if not responses:
            responses.append("I can see your surroundings, but I'm still processing the details.")
        
        return ' '.join(responses)
    
    def _fallback_text_response(self, text: str, intent: str) -> str:
        """Provide immediate, empathetic responses"""
        text_lower = text.lower()
        name = self.user_context.get('name', 'friend')
        
        # Personal greeting responses
        if any(word in text_lower for word in ['hello', 'hi', 'hey']) and 'name' in text_lower:
            if 'diya' in text_lower:
                self.user_context['name'] = 'Diya'
                return "Hello Diya! It's wonderful to meet you. I'm SeeForMe, your AI companion. I can help you understand your surroundings, analyze your emotions, and provide support whenever you need it. I'm here for you, Diya."
            else:
                return f"Hello {name}! I'm SeeForMe, your AI assistant. I'm here to help you with anything you need."
        
        # Emotional support responses
        if any(word in text_lower for word in ['sad', 'feeling sad', 'upset', 'down', 'depressed']):
            return f"I can hear in your voice that you're going through a difficult time, {name}. I'm truly sorry you're feeling sad right now. I want you to know that I'm here to listen and support you. You're not alone. Would you like to tell me more about what's troubling you, or would you prefer if I help you with something to lift your spirits?"
        
        # How do I look requests
        if 'how do i look' in text_lower or 'my face' in text_lower or 'my expression' in text_lower:
            return f"I'd love to help you understand how you look and feel, {name}. Let me switch to the front camera to analyze your expression and mood. This will help me give you better support."
        
        # What do you see requests
        if any(phrase in text_lower for phrase in ['what do you see', 'what\'s there', 'look around', 'surroundings']):
            return f"I'm ready to be your eyes, {name}. Let me switch to the back camera and describe everything I can see around you."
        
        # General conversation
        return f"I hear you, {name}. I'm listening carefully to everything you're telling me. How can I best help you right now? I can describe your surroundings, check your expression, or just talk with you about whatever is on your mind."
    
    def _extract_person_region(self, frame, bbox):
        """Extract person region from frame for emotion analysis"""
        if not bbox or len(bbox) < 4:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # Extract region
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Error extracting person region: {e}")
        
        return None
    
    def _emit_status_update(self):
        """Emit status update to client"""
        status = self.get_status()
        self.socketio.emit('status_update', status)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'status': 'active' if self.is_active else 'ready',
            'listening': self.is_listening,
            'camera_permission': self.camera_permission,
            'components': {
                'speech': self.speech_handler.get_status() if self.speech_handler else {'status': 'not_initialized'},
                'camera': self.camera_switcher.get_status() if self.camera_switcher else {'status': 'not_initialized'},
                'emotion': self.emotion_detector.get_status() if self.emotion_detector else {'status': 'not_initialized'},
                'scene': self.scene_detector.get_status() if self.scene_detector else {'status': 'not_initialized'},
                'gemma': self.gemma_connector.get_status() if self.gemma_connector else {'status': 'not_initialized'},
                'tts': self.tts_handler.get_status() if self.tts_handler else {'status': 'not_initialized'}
            },
            'user_context': self.user_context
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up assistant coordinator...")
        
        self.stop_event.set()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        if self.speech_handler:
            self.speech_handler.cleanup()
        
        if self.camera_switcher:
            self.camera_switcher.cleanup()
        
        if self.tts_handler:
            self.tts_handler.cleanup()
        
        logger.info("✅ Cleanup completed")