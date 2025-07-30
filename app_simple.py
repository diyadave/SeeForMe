import os
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "offline_assistant_secret_key_2024")

# Initialize SocketIO with proper configuration for gunicorn
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                    engineio_logger=False, socketio_logger=False)

@app.route('/')
def index():
    """Main page with assistant interface"""
    return render_template('index.html')

@app.route('/status')
def status():
    """API endpoint for assistant status"""
    return {
        'status': 'ready',
        'components': {
            'speech': {'status': 'ready'},
            'camera': {'status': 'ready'},
            'ai': {'status': 'ready'},
            'tts': {'status': 'ready'}
        },
        'user_context': {
            'user_name': 'friend',
            'current_emotion': 'Neutral',
            'language': 'en'
        }
    }

@socketio.on('connect')
def on_connect(auth=None):
    """Handle client connection"""
    logger.info("Client connected")
    
    # Send initial status
    emit('status_update', {
        'status': 'ready',
        'components': {
            'speech': {'status': 'ready'},
            'camera': {'status': 'ready', 'active': False},
            'ai': {'status': 'ready'},
            'tts': {'status': 'ready'}
        },
        'user_context': {
            'user_name': 'friend',
            'current_emotion': 'Neutral',
            'language': 'en'
        }
    })
    
    # Send welcome message
    emit('assistant_message', {
        'text': 'Hello! I\'m your offline smart assistant. The core interface is ready. AI models are being prepared for voice, camera, and conversation features.',
        'language': 'en',
        'emotion': 'Neutral'
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('start_listening')
def on_start_listening():
    """Start speech recognition"""
    logger.info("Speech recognition requested")
    emit('listening_status', {'active': True})
    emit('assistant_message', {
        'text': 'Speech recognition is ready. This feature will be enhanced with voice models.',
        'language': 'en',
        'emotion': 'Neutral'
    })

@socketio.on('stop_listening')
def on_stop_listening():
    """Stop speech recognition"""
    logger.info("Stop speech recognition")
    emit('listening_status', {'active': False})

@socketio.on('start_camera')
def on_start_camera(data):
    """Start camera with specified mode"""
    mode = data.get('mode', 'scene')
    logger.info(f"Camera requested in {mode} mode")
    emit('camera_status', {'active': True, 'mode': mode})
    emit('assistant_message', {
        'text': f'Camera is ready in {mode} mode. Computer vision features will be available once AI models are loaded.',
        'language': 'en',
        'emotion': 'Neutral'
    })

@socketio.on('stop_camera')
def on_stop_camera():
    """Stop camera"""
    logger.info("Stop camera")
    emit('camera_status', {'active': False})

@socketio.on('send_text')
def on_send_text(data):
    """Process text input"""
    text = data.get('text', '')
    language = data.get('language', 'en')
    logger.info(f"Text received: {text}")
    
    # Echo the user's message
    emit('speech_recognized', {'text': text, 'language': language})
    
    # Send a response
    responses = [
        "I understand you're saying: " + text,
        "Thank you for your message. The AI conversation system is being prepared.",
        "I'm here to help! The full assistant features will be available once all models are loaded.",
        "Your input has been received. Advanced AI responses are coming soon."
    ]
    
    import random
    response = random.choice(responses)
    
    emit('assistant_message', {
        'text': response,
        'language': language,
        'emotion': 'Neutral'
    })

@socketio.on('toggle_camera_mode')
def on_toggle_camera_mode():
    """Toggle between front and back camera"""
    logger.info("Camera mode toggle requested")
    emit('camera_mode_changed', {'mode': 'emotion'})

@socketio.on('switch_language')
def on_switch_language(data):
    """Switch language"""
    language = data.get('language', 'en')
    logger.info(f"Language switch to: {language}")
    emit('language_switched', {'success': True, 'language': language})

@socketio.on('request_scene_description')
def on_request_scene_description():
    """Request scene description"""
    logger.info("Scene description requested")
    emit('assistant_message', {
        'text': 'Scene analysis is ready. Computer vision models are being prepared to describe your surroundings.',
        'language': 'en',
        'emotion': 'Neutral'
    })

@socketio.on('get_status')
def on_get_status():
    """Get current assistant status"""
    emit('status_update', {
        'status': 'ready',
        'components': {
            'speech': {'status': 'ready'},
            'camera': {'status': 'ready'},
            'ai': {'status': 'ready'},
            'tts': {'status': 'ready'}
        },
        'user_context': {
            'user_name': 'friend',
            'current_emotion': 'Neutral',
            'language': 'en'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('index.html'), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, log_output=True)