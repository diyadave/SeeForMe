#!/usr/bin/env python3
"""
Simple Voice Assistant - Fallback Implementation
Direct voice recognition and response without complex architecture
"""

import os
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "simple_voice_assistant")

# Simple SocketIO setup - no complex threading
socketio = SocketIO(app, 
                    cors_allowed_origins="*",
                    logger=False,
                    engineio_logger=False)

@app.route('/')
def index():
    """Simple voice assistant interface"""
    return render_template('simple_voice.html')

@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    logger.info("✅ Client connected")
    emit('connected', {'message': 'Voice assistant ready'})

@socketio.on('speech_recognized')
def on_speech_recognized(data):
    """Handle speech recognition - guaranteed to work"""
    try:
        text = data.get('text', '').lower()
        logger.info(f"🗣️ Speech received: {text}")
        
        # Simple response logic
        if 'hello' in text or 'hi' in text:
            response = "Hello! I'm your voice assistant. How can I help you today?"
        elif 'feeling bad' in text or 'sad' in text or 'tired' in text:
            response = f"I'm sorry to hear you're feeling this way. I'm here to listen and support you. Would you like to talk about what's bothering you?"
        elif 'name' in text:
            if 'diya' in text.lower():
                response = "Nice to meet you, Diya! I'm your voice assistant and I'm here to help you."
            else:
                response = "It's great to meet you! I'm your voice assistant."
        else:
            response = f"I heard you say: '{text}'. I'm listening and ready to help you with anything you need."
        
        # Send response immediately
        emit('assistant_response', {
            'text': response,
            'speak': True
        })
        
        logger.info(f"✅ Response sent: {response[:50]}...")
        
    except Exception as e:
        logger.error(f"❌ Error processing speech: {e}")
        emit('assistant_response', {
            'text': 'I heard you, but had trouble processing your request. Please try again.',
            'speak': True
        })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnect"""
    logger.info("❌ Client disconnected")

if __name__ == '__main__':
    logger.info("🚀 Starting Simple Voice Assistant")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)