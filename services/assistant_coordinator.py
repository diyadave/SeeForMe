#!/usr/bin/env python3
"""
Assistant Coordinator - Orchestrates all components
Manages threading, communication, and state between all services
"""

import logging
import threading
import queue
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .gemma_connect import GemmaConnector
from .name_extractor import NameExtractor
from .scene_detector import SceneDetector
from .speech_handler import VoskSpeechRecognizer
from .tts_handler import TTSHandler

logger = logging.getLogger(__name__)

class AssistantCoordinator:
    """Coordinates all assistant components with proper threading"""
    
    def __init__(self, socketio=None):
        logger.info("🤖 Initializing Assistant Coordinator...")
        
        self.socketio = socketio
        
        # Component initialization
        self.gemma = None
        self.name_extractor = None
        self.scene_detector = None
        self.speech_handler = None
        self.tts_handler = None
        
        # Camera management
        self.camera = None
        self.camera_thread = None
        self.camera_active = False
        self.camera_mode = 'scene'  # 'scene' or 'emotion'
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # User context
        self.user_context = {
            'user_name': 'friend',
            'current_emotion': 'Neutral',
            'scene_info': 'indoor space',
            'language': 'en',
            'last_interaction': time.time()
        }
        
        # Communication queues
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Threading control
        self.running = True
        self.coordinator_thread = None
        
        # Performance tracking
        self.interaction_count = 0
        self.successful_interactions = 0
        
        # Initialize components
        self.initialize_components()
        
        # Start coordinator thread
        self.start_coordinator()
        
        logger.info("✅ Assistant Coordinator initialized")
    
    def initialize_components(self):
        """Initialize all assistant components"""
        try:
            # Initialize Gemma connector
            logger.info("🧠 Initializing Gemma connector...")
            self.gemma = GemmaConnector()
            
            # Initialize name extractor
            logger.info("👤 Initializing name extractor...")
            self.name_extractor = NameExtractor()
            
            # Initialize scene detector
            logger.info("👁️ Initializing scene detector...")
            self.scene_detector = SceneDetector()
            
            # Initialize speech handler with callback
            logger.info("🎤 Initializing speech handler...")
            self.speech_handler = VoskSpeechRecognizer(callback=self.on_speech_recognized)
            
            # Initialize TTS handler
            logger.info("🔊 Initializing TTS handler...")
            self.tts_handler = TTSHandler()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def start_coordinator(self):
        """Start the main coordinator thread"""
        self.coordinator_thread = threading.Thread(target=self.coordinator_worker, daemon=True)
        self.coordinator_thread.start()
        logger.info("🎛️ Coordinator thread started")
    
    def coordinator_worker(self):
        """Main coordinator loop"""
        logger.info("🎛️ Coordinator worker started")
        
        while self.running:
            try:
                # Process messages from queue
                try:
                    message = self.message_queue.get(timeout=1)
                    self.process_message(message)
                except queue.Empty:
                    continue
                
                # Update scene analysis if camera is active
                if self.camera_active and self.current_frame is not None:
                    self.update_scene_analysis()
                
                # Check for stale interactions
                self.check_interaction_timeout()
                
            except Exception as e:
                logger.error(f"❌ Coordinator worker error: {e}")
                time.sleep(1)
        
        logger.info("🎛️ Coordinator worker stopped")
    
    def process_message(self, message: Dict[str, Any]):
        """Process incoming messages"""
        message_type = message.get('type')
        data = message.get('data', {})
        
        try:
            if message_type == 'user_input':
                self.handle_user_input(data['text'], data.get('language', 'en'))
            elif message_type == 'scene_request':
                self.handle_scene_request()
            elif message_type == 'emotion_check':
                self.handle_emotion_check()
            elif message_type == 'camera_switch':
                self.handle_camera_switch()
            else:
                logger.warning(f"⚠️ Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message {message_type}: {e}")
    
    def handle_user_input(self, text: str, language: str = 'en'):
        """Handle user text input"""
        try:
            self.interaction_count += 1
            self.user_context['last_interaction'] = time.time()
            self.user_context['language'] = language
            
            logger.info(f"💬 User input [{language}]: {text}")
            
            # Extract name if provided
            extracted_name = self.name_extractor.extract_name(text, language)
            if extracted_name:
                self.user_context['user_name'] = extracted_name
                greeting = self.name_extractor.get_greeting_response(extracted_name, language)
                self.send_response(greeting, language)
                return
            
            # Update scene information
            self.update_context_from_camera()
            
            # Generate response using Gemma
            response = self.gemma.generate_response(text, self.user_context)
            
            if response:
                self.send_response(response, language)
                self.successful_interactions += 1
            else:
                fallback_response = "I'm having trouble understanding right now. Could you please repeat that?"
                self.send_response(fallback_response, language)
            
        except Exception as e:
            logger.error(f"❌ Error handling user input: {e}")
            self.send_response("I'm experiencing some technical difficulties. Please try again.", language)
    
    def handle_scene_request(self):
        """Handle scene description request"""
        try:
            if not self.camera_active or self.current_frame is None:
                response = "I need the camera to be active to describe what I see."
                self.send_response(response, self.user_context['language'])
                return
            
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            description = self.scene_detector.get_scene_description(frame)
            self.send_response(description, self.user_context['language'])
            
        except Exception as e:
            logger.error(f"❌ Error handling scene request: {e}")
            self.send_response("I'm having trouble analyzing the scene right now.", self.user_context['language'])
    
    def handle_emotion_check(self):
        """Handle emotion detection request"""
        try:
            if not self.camera_active or self.current_frame is None:
                response = "I need the camera to be active to read emotions."
                self.send_response(response, self.user_context['language'])
                return
            
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            description = self.scene_detector.get_emotion_description(frame)
            
            # Update user context with detected emotion
            emotion_info = self.scene_detector.detect_emotion(frame)
            if emotion_info:
                self.user_context['current_emotion'] = emotion_info.get('emotion', 'Neutral')
            
            self.send_response(description, self.user_context['language'])
            
        except Exception as e:
            logger.error(f"❌ Error handling emotion check: {e}")
            self.send_response("I'm having trouble reading emotions right now.", self.user_context['language'])
    
    def handle_camera_switch(self):
        """Handle camera mode switching"""
        new_mode = 'emotion' if self.camera_mode == 'scene' else 'scene'
        self.camera_mode = new_mode
        
        mode_text = "emotion detection" if new_mode == 'emotion' else "scene analysis"
        response = f"Switched camera to {mode_text} mode."
        self.send_response(response, self.user_context['language'])
        
        # Notify client
        if self.socketio:
            self.socketio.emit('camera_mode_changed', {'mode': new_mode})
    
    def update_scene_analysis(self):
        """Update scene analysis from current frame"""
        try:
            if self.current_frame is None:
                return
            
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            # Analyze based on current mode
            if self.camera_mode == 'scene':
                scene_info = self.scene_detector.classify_scene(frame)
                self.user_context['scene_info'] = scene_info.get('scene_type', 'indoor space')
            elif self.camera_mode == 'emotion':
                emotion_info = self.scene_detector.detect_emotion(frame)
                if emotion_info:
                    self.user_context['current_emotion'] = emotion_info.get('emotion', 'Neutral')
            
        except Exception as e:
            logger.debug(f"Scene analysis update error: {e}")
    
    def update_context_from_camera(self):
        """Update user context from camera if available"""
        if not self.camera_active or self.current_frame is None:
            return
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            # Get scene information
            scene_info = self.scene_detector.classify_scene(frame)
            self.user_context['scene_info'] = scene_info.get('scene_type', 'indoor space')
            
            # Get emotion information
            emotion_info = self.scene_detector.detect_emotion(frame)
            if emotion_info:
                self.user_context['current_emotion'] = emotion_info.get('emotion', 'Neutral')
            
        except Exception as e:
            logger.debug(f"Context update error: {e}")
    
    def send_response(self, text: str, language: str = 'en'):
        """Send response via TTS and socket"""
        try:
            logger.info(f"🗣️ Sending response [{language}]: {text}")
            
            # Send via TTS
            self.tts_handler.speak(text, language)
            
            # Send via socket
            if self.socketio:
                self.socketio.emit('assistant_message', {
                    'text': text,
                    'language': language,
                    'emotion': self.user_context.get('current_emotion', 'Neutral'),
                    'timestamp': time.time()
                })
            
        except Exception as e:
            logger.error(f"❌ Error sending response: {e}")
    
    def on_speech_recognized(self, result: Dict[str, Any]):
        """Callback for speech recognition"""
        try:
            text = result.get('text', '').strip()
            language = result.get('language', 'en')
            confidence = result.get('confidence', 0.0)
            
            if text and confidence > 0.5:
                logger.info(f"🎤 Speech recognized: '{text}' (conf: {confidence:.2f})")
                
                # Add to message queue
                self.message_queue.put({
                    'type': 'user_input',
                    'data': {'text': text, 'language': language}
                })
                
                # Notify client
                if self.socketio:
                    self.socketio.emit('speech_recognized', {
                        'text': text,
                        'language': language,
                        'confidence': confidence
                    })
            
        except Exception as e:
            logger.error(f"❌ Error processing speech recognition: {e}")
    
    def start_camera(self, mode: str = 'scene') -> bool:
        """Start camera capture"""
        try:
            if self.camera_active:
                logger.info("📹 Camera already active")
                return True
            
            self.camera_mode = mode
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                logger.error("❌ Failed to open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            
            self.camera_active = True
            self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
            self.camera_thread.start()
            
            logger.info(f"📹 Camera started in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        try:
            if not self.camera_active:
                return
            
            self.camera_active = False
            
            if self.camera_thread:
                self.camera_thread.join(timeout=2)
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            with self.frame_lock:
                self.current_frame = None
            
            logger.info("📹 Camera stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping camera: {e}")
    
    def camera_worker(self):
        """Camera capture worker"""
        logger.info("📹 Camera worker started")
        
        while self.camera_active and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame
                else:
                    logger.warning("⚠️ Failed to read camera frame")
                    time.sleep(0.1)
                
                # Control frame rate
                time.sleep(1/15)  # 15 FPS
                
            except Exception as e:
                logger.error(f"❌ Camera worker error: {e}")
                time.sleep(1)
        
        logger.info("📹 Camera worker stopped")
    
    def toggle_camera_mode(self) -> str:
        """Toggle between scene and emotion modes"""
        new_mode = 'emotion' if self.camera_mode == 'scene' else 'scene'
        self.camera_mode = new_mode
        logger.info(f"📷 Camera mode switched to {new_mode}")
        return new_mode
    
    def start_listening(self) -> bool:
        """Start speech recognition"""
        if self.speech_handler and self.speech_handler.available:
            return self.speech_handler.start_listening()
        return False
    
    def stop_listening(self):
        """Stop speech recognition"""
        if self.speech_handler:
            self.speech_handler.stop_listening()
    
    def switch_language(self, language: str) -> bool:
        """Switch language mode"""
        try:
            success = True
            
            # Switch speech recognition language
            if self.speech_handler:
                success &= self.speech_handler.switch_language(language)
            
            # Update context
            self.user_context['language'] = language
            
            logger.info(f"🌐 Language switched to {language}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error switching language: {e}")
            return False
    
    def process_user_input(self, text: str, language: str = 'en'):
        """Process user text input (external interface)"""
        self.message_queue.put({
            'type': 'user_input',
            'data': {'text': text, 'language': language}
        })
    
    def describe_current_scene(self):
        """Request current scene description"""
        self.message_queue.put({'type': 'scene_request', 'data': {}})
    
    def check_interaction_timeout(self):
        """Check for interaction timeout and send reminders"""
        try:
            time_since_last = time.time() - self.user_context['last_interaction']
            
            # Send reminder after 5 minutes of inactivity
            if time_since_last > 300:  # 5 minutes
                reminder = "I'm still here if you need assistance. Just say something or ask me to describe what I see."
                self.send_response(reminder, self.user_context['language'])
                self.user_context['last_interaction'] = time.time()  # Reset timer
                
        except Exception as e:
            logger.error(f"❌ Error checking interaction timeout: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        try:
            status = {
                'coordinator': {
                    'running': self.running,
                    'interaction_count': self.interaction_count,
                    'success_rate': f"{(self.successful_interactions / self.interaction_count * 100):.1f}%" if self.interaction_count > 0 else "0%"
                },
                'camera': {
                    'active': self.camera_active,
                    'mode': self.camera_mode,
                    'has_frame': self.current_frame is not None
                },
                'user_context': self.user_context.copy(),
                'components': {}
            }
            
            # Add component statuses
            if self.gemma:
                status['components']['gemma'] = self.gemma.get_status()
            
            if self.name_extractor:
                status['components']['name_extractor'] = self.name_extractor.get_status()
            
            if self.scene_detector:
                status['components']['scene_detector'] = self.scene_detector.get_status()
            
            if self.speech_handler:
                status['components']['speech_handler'] = self.speech_handler.get_status()
            
            if self.tts_handler:
                status['components']['tts_handler'] = self.tts_handler.get_status()
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up all resources"""
        try:
            logger.info("🧹 Starting assistant cleanup...")
            
            self.running = False
            
            # Stop camera
            self.stop_camera()
            
            # Stop speech recognition
            self.stop_listening()
            
            # Wait for coordinator thread
            if self.coordinator_thread and self.coordinator_thread.is_alive():
                self.coordinator_thread.join(timeout=3)
            
            # Cleanup components
            if self.gemma:
                self.gemma.cleanup()
            
            if self.scene_detector:
                self.scene_detector.cleanup()
            
            if self.speech_handler:
                self.speech_handler.cleanup()
            
            if self.tts_handler:
                self.tts_handler.cleanup()
            
            # Clear queues
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("✅ Assistant cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Test function
if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        assistant.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    assistant = AssistantCoordinator()
    
    try:
        # Start camera and listening
        assistant.start_camera('scene')
        assistant.start_listening()
        
        # Test interaction
        assistant.process_user_input("Hello, my name is Alex")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        assistant.cleanup()
