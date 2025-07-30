import os
import logging
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import threading
import queue
import time
import json
from pathlib import Path

# Import our services
from services.assistant_coordinator import AssistantCoordinator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "offline_assistant_secret_key_2024")

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global assistant coordinator
assistant = None

@app.route('/')
def index():
    """Main page with assistant interface"""
    return render_template('index.html')

@app.route('/status')
def status():
    """API endpoint for assistant status"""
    if assistant:
        return assistant.get_status()
    return {'status': 'not_initialized'}

@socketio.on('connect')
def on_connect(auth=None):
    """Handle client connection"""
    logger.info("Client connected")
    
    # Initialize assistant if not already done
    global assistant
    if assistant is None:
        try:
            assistant = AssistantCoordinator(socketio)
            logger.info("✅ Assistant coordinator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize assistant: {e}")
            emit('error', {'message': 'Failed to initialize assistant'})
            return
    
    # Send initial status
    emit('status_update', assistant.get_status())
    
    # Send welcome message
    emit('assistant_message', {
        'text': 'Hello! I\'m your offline smart assistant. I can see through your camera, listen to your voice, and help you understand your surroundings. How can I assist you today?',
        'language': 'en',
        'emotion': 'Neutral'
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('start_camera')
def on_start_camera(data):
    """Start camera with specified mode"""
    if assistant:
        camera_mode = data.get('mode', 'scene')  # 'scene' or 'emotion'
        success = assistant.start_camera(camera_mode)
        emit('camera_status', {'active': success, 'mode': camera_mode})
    else:
        emit('error', {'message': 'Assistant not initialized'})

@socketio.on('stop_camera')
def on_stop_camera():
    """Stop camera"""
    if assistant:
        assistant.stop_camera()
        emit('camera_status', {'active': False})

@socketio.on('start_listening')
def on_start_listening():
    """Start speech recognition"""
    if assistant:
        success = assistant.start_listening()
        emit('listening_status', {'active': success})
    else:
        emit('error', {'message': 'Assistant not initialized'})

@socketio.on('stop_listening')
def on_stop_listening():
    """Stop speech recognition"""
    if assistant:
        assistant.stop_listening()
        emit('listening_status', {'active': False})

@socketio.on('send_text')
def on_send_text(data):
    """Process text input from user"""
    if assistant:
        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        if text:
            assistant.process_user_input(text, language)
    else:
        emit('error', {'message': 'Assistant not initialized'})

@socketio.on('switch_language')
def on_switch_language(data):
    """Switch language mode"""
    if assistant:
        language = data.get('language', 'en')
        success = assistant.switch_language(language)
        emit('language_switched', {'language': language, 'success': success})

@socketio.on('request_scene_description')
def on_request_scene_description():
    """Request current scene description"""
    if assistant:
        assistant.describe_current_scene()

@socketio.on('toggle_camera_mode')
def on_toggle_camera_mode():
    """Toggle between front and back camera"""
    if assistant:
        new_mode = assistant.toggle_camera_mode()
        emit('camera_mode_changed', {'mode': new_mode})

@socketio.on('get_status')
def on_get_status():
    """Get current assistant status"""
    if assistant:
        emit('status_update', assistant.get_status())
    else:
        emit('status_update', {'status': 'not_initialized'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('index.html'), 500

# Cleanup on shutdown
import atexit

def cleanup():
    """Cleanup resources on shutdown"""
    global assistant
    if assistant:
        logger.info("🧹 Cleaning up assistant resources...")
        assistant.cleanup()

atexit.register(cleanup)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, log_output=True)
