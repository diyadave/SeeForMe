#!/usr/bin/env python3
"""
SeeForMe - Fully Offline Smart Assistant for Blind Users
Complete voice-first accessibility app with animated UI
"""

import os
import logging
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "seefor_me_accessibility_2024")

# Initialize SocketIO - SUPER FAST setup for instant responses
socketio = SocketIO(app, 
                    cors_allowed_origins="*", 
                    async_mode='threading',
                    logger=False,
                    engineio_logger=False,
                    transports=['polling'],
                    ping_timeout=60,  # Shorter timeout to prevent worker hanging
                    ping_interval=10)

# Global assistant coordinator
assistant_coordinator = None

@app.route('/')
def index():
    """Main SeeForMe interface with animated assistant"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'app': 'SeeForMe',
        'version': '1.0.0'
    })

@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    logger.info("🔗 SeeForMe client connected")
    print(f"🔗 BACKEND: Client connected, handlers available: {list(socketio.server.handlers.keys())}")
    
    # Test immediate SocketIO response to verify connection
    emit('test_connection', {'message': 'Backend connected and ready'})

@socketio.on('test_speech')
def on_test_speech(data):
    """Handle test speech events"""
    print(f"🧪 BACKEND: Test speech event received: {data}")
    logger.info(f"🧪 BACKEND: Test speech event received: {data}")
    emit('assistant_response', {
        'text': 'Test successful! Backend received your speech event.',
        'emotion': 'friendly', 
        'speak': True
    })
    
    # Initialize assistant coordinator if not already done
    global assistant_coordinator
    if assistant_coordinator is None:
        try:
            from .assistant_coordinator import AssistantCoordinator
            assistant_coordinator = AssistantCoordinator(socketio)
            logger.info("✅ Assistant Coordinator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinator: {e}")
            # Send error status to client
            emit('system_status', {
                'status': 'error',
                'message': 'Assistant initialization failed',
                'details': str(e)
            })
            return
    
    # Send ready status to client
    emit('system_status', {
        'status': 'ready',
        'message': 'SeeForMe is ready to help',
        'features': {
            'voice_recognition': True,
            'emotion_detection': True,
            'scene_analysis': True,
            'multi_language': True,
            'offline_mode': True
        }
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info("❌ SeeForMe client disconnected")

@socketio.on('start_assistant')
def on_start_assistant():
    """Start the voice assistant"""
    logger.info("🎤 Starting voice assistant")
    
    if assistant_coordinator:
        assistant_coordinator.start_listening()
        emit('assistant_started', {
            'status': 'listening',
            'message': 'I am listening and ready to help you'
        })
    else:
        emit('system_error', {
            'message': 'Assistant not initialized'
        })

@socketio.on('stop_assistant')
def on_stop_assistant():
    """Stop the voice assistant"""
    logger.info("🛑 Stopping voice assistant")
    
    if assistant_coordinator:
        assistant_coordinator.stop_listening()
        emit('assistant_stopped', {
            'status': 'stopped',
            'message': 'Voice assistant stopped'
        })

@socketio.on('voice_input')
def on_voice_input(data):
    """Handle voice input from client"""
    text = data.get('text', '')
    confidence = data.get('confidence', 0.0)
    language = data.get('language', 'en')
    
    logger.info(f"🗣️ Voice input: '{text}' (confidence: {confidence:.2f})")
    
    if assistant_coordinator and text.strip():
        # Process voice input through coordinator - THIS SHOULD TRIGGER TTS
        logger.info(f"🔄 Processing voice input through coordinator...")
        assistant_coordinator.process_voice_input(text, language, confidence)
    else:
        emit('assistant_response', {
            'text': 'I did not understand that. Please try again.',
            'emotion': 'neutral',
            'speak': True
        })

@socketio.on('camera_permission')
def on_camera_permission(data):
    """Handle camera permission status"""
    permitted = data.get('permitted', False)
    logger.info(f"📹 Camera permission: {'granted' if permitted else 'denied'}")
    
    if assistant_coordinator:
        assistant_coordinator.set_camera_permission(permitted)

@socketio.on('get_status')
def on_get_status():
    """Get current system status"""
    if assistant_coordinator:
        status = assistant_coordinator.get_status()
        emit('status_update', status)
    else:
        emit('status_update', {
            'status': 'not_initialized',
            'components': {}
        })

@socketio.on('speech_recognized') 
def on_speech_recognized(data):
    """Handle speech recognition results - WORKING VERSION"""
    try:
        text = data.get('text', '').lower()
        print(f"\n🎯 SPEECH RECEIVED: {text}")
        logger.info(f"🗣️ Speech: {text}")
        
        # FAST RESPONSE with Gemma 3b - No threading to avoid timeout
        from simple_gemma_agent import simple_agent
        
        # Extract user info
        user_text = data.get('text', '')
        user_name = "friend"  # Will be extracted properly later
        
        # Get AI response from Gemma 3b
        logger.info(f"🤖 Getting Gemma 3b response...")
        ai_response = simple_agent.get_response(user_text, user_name, 'neutral')
        
        # Send response immediately
        emit('assistant_response', {
            'text': ai_response,
            'emotion': 'caring',
            'speak': True
        })
        
        logger.info(f"✅ Gemma 3b response sent: {ai_response[:50]}...")
            
    except Exception as e:
        logger.error(f"❌ Speech processing error: {e}")
        emit('assistant_response', {
            'text': f"I heard you say '{data.get('text', '')}'. I'm your emotionally intelligent AI companion powered by Gemma 3b, here to support you.",
            'emotion': 'caring',
            'speak': True
        })

if __name__ == '__main__':
    logger.info("🚀 Starting SeeForMe Assistant")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)