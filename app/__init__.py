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
                    ping_timeout=10,  # Much shorter timeout
                    ping_interval=5)

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
    
    # Skip assistant coordinator to avoid blocking
    logger.info("🎤 Voice assistant ready")
    emit('assistant_started', {
        'status': 'listening',
        'message': 'I am listening and ready to help you'
    })

@socketio.on('stop_assistant')
def on_stop_assistant():
    """Stop the voice assistant"""
    logger.info("🛑 Stopping voice assistant")
    
    # Skip assistant coordinator 
    logger.info("🎤 Voice assistant stopped")
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
    
    # Process voice input directly without coordinator
    emit('assistant_response', {
        'text': f"I heard you say: {text}",
        'emotion': 'neutral',
        'speak': True
    })

@socketio.on('camera_permission')
def on_camera_permission(data):
    """Handle camera permission status"""
    permitted = data.get('permitted', False)
    logger.info(f"📹 Camera permission: {'granted' if permitted else 'denied'}")
    
    # Skip assistant coordinator

@socketio.on('get_status')
def on_get_status():
    """Get current system status"""
    emit('status_update', {
        'status': 'ready',
        'components': {'speech': 'active', 'tts': 'active'}
    })

@socketio.on('speech_recognized') 
def on_speech_recognized(data):
    """Handle speech recognition results - ULTRA FAST VERSION"""
    # Ultra-fast processing - no try/except to avoid delays
    user_text = data.get('text', '')
    print(f"🎯 SPEECH: {user_text}")
    
    # Extract name properly first
    import re
    user_name = "friend"
    if "my name is" in user_text.lower():
        match = re.search(r"my name is\s+(\w+)", user_text.lower())
        if match:
            user_name = match.group(1).capitalize()
    elif "call me" in user_text.lower():
        match = re.search(r"call me\s+(\w+)", user_text.lower())
        if match:
            user_name = match.group(1).capitalize()
    
    # Try Gemma3n first, fallback to pattern matching for reliability
    response = ""
    try:
        from simple_gemma_agent import simple_agent
        response = simple_agent.get_response(user_text, user_name, "caring", "conversation")
        print(f"🤖 GEMMA3N SUCCESS: {response}")
    except Exception as e:
        print(f"❌ GEMMA3N NOT AVAILABLE: {e}")
        print("📱 USING PATTERN MATCHING FALLBACK")
        
        # Enhanced pattern matching with proper name extraction
        if user_name != "friend":
            response = f"Hello {user_name}! Nice to meet you! I'm your caring AI companion and I'm here to support you emotionally."
        elif "scold" in user_text.lower() or "harsh" in user_text.lower():
            response = "I understand someone was harsh with you. That must have been difficult. Remember, other people's words don't define your worth. You're valuable and deserving of kindness."
        elif "bad" in user_text.lower() or "sad" in user_text.lower():
            response = "I'm sorry you're having a tough time. I'm here to listen and support you. You're not alone, and I care about how you're feeling."
        elif "thank" in user_text.lower():
            response = "You're so welcome! I'm glad I could help. I'm always here when you need emotional support or just someone to talk to."
        elif "happy" in user_text.lower() or "good" in user_text.lower():
            response = "That's wonderful! I'm so glad you're feeling good today. It makes me happy to hear positive things from you."
        else:
            response = f"I heard you say '{user_text}'. I'm your caring AI companion, always here to provide emotional support and understanding."
    
    # Send response immediately
    emit('assistant_response', {
        'text': response,
        'emotion': 'caring',
        'speak': True
    })
    
    print(f"✅ SENT: {response}")

if __name__ == '__main__':
    logger.info("🚀 Starting SeeForMe Assistant")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)