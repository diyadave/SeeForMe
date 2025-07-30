import os
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import threading
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "offline_assistant_secret_key_2024")

# Initialize SocketIO with proper configuration
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                    engineio_logger=False, socketio_logger=False)

# Global assistant coordinator - will be initialized with full AI stack
assistant = None

@app.route('/')
def index():
    """Voice-first assistant interface"""
    return render_template('voice_assistant.html')

@app.route('/status')
def status():
    """API endpoint for assistant status"""
    if assistant:
        return assistant.get_status()
    return {
        'status': 'initializing',
        'components': {
            'speech': {'status': 'ready', 'listening': True},
            'camera': {'status': 'ready', 'auto_mode': True},
            'ai': {'status': 'loading_gemma'},
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
    """Handle client connection - Auto-start everything for accessibility"""
    logger.info("Voice-first assistant client connected")
    
    # Initialize assistant with full AI stack
    global assistant
    if assistant is None:
        try:
            # Import and initialize the full assistant coordinator
            from services.assistant_coordinator import AssistantCoordinator
            assistant = AssistantCoordinator(socketio)
            logger.info("✅ Full AI assistant coordinator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize full assistant: {e}")
            # Fall back to basic functionality
            assistant = None
    
    # Send initial status
    emit('status_update', {
        'status': 'ready',
        'voice_activated': True,
        'auto_mode': True,
        'components': {
            'speech': {'status': 'listening', 'active': True},
            'camera': {'status': 'ready', 'auto_switching': True},
            'ai': {'status': 'gemma_ready'},
            'tts': {'status': 'ready'}
        }
    })
    
    # Auto-start voice recognition for accessibility
    emit('auto_start_listening')
    
    # Welcome message with instructions
    emit('assistant_message', {
        'text': 'Hello! I am SeeForMe, your intelligent vision assistant. I am listening and ready to help. Just speak naturally - I will understand what you need and automatically switch between analyzing your surroundings or checking your emotions. How can I assist you today?',
        'language': 'en',
        'emotion': 'Friendly',
        'auto_speak': True
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info("Voice assistant client disconnected")

@socketio.on('voice_input')
def on_voice_input(data):
    """Process voice input and intelligently decide what to do"""
    text = data.get('text', '').lower()
    language = data.get('language', 'en')
    confidence = data.get('confidence', 0.0)
    
    logger.info(f"Voice input received: '{text}' (confidence: {confidence})")
    
    # Emit that we heard the user
    emit('speech_recognized', {'text': text, 'language': language, 'confidence': confidence})
    
    # Intelligent intent detection
    intent = detect_user_intent(text)
    
    if assistant:
        # Use full AI assistant
        response = assistant.process_voice_input(text, language, intent)
    else:
        # Fallback processing
        response = process_fallback_response(text, intent, language)
    
    # Send response
    emit('assistant_message', {
        'text': response['text'],
        'language': response['language'],
        'emotion': response.get('emotion', 'Neutral'),
        'auto_speak': True,
        'intent': intent
    })
    
    # Auto-trigger camera if needed
    if intent in ['scene_description', 'environment_check', 'what_see']:
        emit('auto_camera_scene')
    elif intent in ['emotion_check', 'how_look', 'my_mood']:
        emit('auto_camera_emotion')

def detect_user_intent(text):
    """Intelligent intent detection from voice input"""
    text = text.lower()
    
    # Scene/Environment intents
    scene_keywords = [
        'what do you see', 'describe surroundings', 'what\'s around me', 
        'describe the scene', 'what\'s in front', 'environment', 'room',
        'where am i', 'what\'s here', 'look around', 'describe what you see',
        'what is around', 'tell me about', 'surroundings'
    ]
    
    # Emotion/Appearance intents  
    emotion_keywords = [
        'how do i look', 'what\'s my mood', 'how am i feeling', 'my emotion',
        'do i look', 'my face', 'how do i appear', 'my expression',
        'am i smiling', 'do i look happy', 'how is my face'
    ]
    
    # General conversation
    conversation_keywords = [
        'hello', 'hi', 'help', 'how are you', 'what can you do',
        'tell me about yourself', 'good morning', 'good evening'
    ]
    
    # Check for specific intents
    for keyword in scene_keywords:
        if keyword in text:
            return 'scene_description'
    
    for keyword in emotion_keywords:
        if keyword in text:
            return 'emotion_check'
    
    for keyword in conversation_keywords:
        if keyword in text:
            return 'general_conversation'
    
    # Default: if short, likely conversation; if longer, might want scene description
    if len(text.split()) <= 3:
        return 'general_conversation'
    else:
        return 'scene_description'

def process_fallback_response(text, intent, language):
    """Fallback response processing when full AI is not available"""
    
    responses = {
        'scene_description': [
            "I'm preparing to analyze your surroundings. The camera system is getting ready to describe what's around you.",
            "Let me look around for you. I'm activating the scene analysis to tell you about your environment.",
            "I'll describe what I can see. The vision system is processing your surroundings now."
        ],
        'emotion_check': [
            "I'm checking how you're feeling by looking at your expression. The emotion detection is activating.",
            "Let me see your face to understand your current mood. Switching to emotion analysis mode.",
            "I'll analyze your facial expression to tell you about your current emotional state."
        ],
        'general_conversation': [
            f"I understand you said: {text}. I'm here to help you see and understand your world.",
            "Thank you for talking with me. I can describe your surroundings or check your emotions - just ask!",
            "I'm listening and ready to assist. I can analyze what's around you or how you're feeling."
        ]
    }
    
    import random
    response_text = random.choice(responses.get(intent, responses['general_conversation']))
    
    return {
        'text': response_text,
        'language': language,
        'emotion': 'Helpful'
    }

@socketio.on('camera_analysis_result')
def on_camera_analysis_result(data):
    """Handle camera analysis results and provide intelligent response"""
    analysis_type = data.get('type', 'scene')
    results = data.get('results', {})
    
    if analysis_type == 'scene':
        description = generate_scene_description(results)
    elif analysis_type == 'emotion':
        description = generate_emotion_description(results)
    else:
        description = "I've completed the analysis."
    
    emit('assistant_message', {
        'text': description,
        'language': 'en',
        'emotion': 'Informative',
        'auto_speak': True
    })

def generate_scene_description(results):
    """Generate natural scene description from analysis results"""
    if not results:
        return "I'm having difficulty seeing clearly right now. Please make sure the camera has a good view."
    
    # Extract information
    objects = results.get('objects', [])
    scene_type = results.get('scene_type', 'indoor space')
    people_count = results.get('people_count', 0)
    
    description_parts = []
    
    # Scene context
    description_parts.append(f"You appear to be in {scene_type}.")
    
    # People
    if people_count > 0:
        if people_count == 1:
            description_parts.append("There is one person visible.")
        else:
            description_parts.append(f"I can see {people_count} people.")
    
    # Objects
    if objects:
        if len(objects) <= 3:
            obj_list = ", ".join(objects)
            description_parts.append(f"I can see: {obj_list}.")
        else:
            description_parts.append(f"I can see several items including {objects[0]}, {objects[1]}, and {len(objects)-2} other objects.")
    
    return " ".join(description_parts)

def generate_emotion_description(results):
    """Generate natural emotion description from analysis results"""
    if not results:
        return "I cannot clearly see your face right now. Please face the camera so I can check your expression."
    
    emotion = results.get('emotion', 'Neutral')
    confidence = results.get('confidence', 0.0)
    
    emotion_descriptions = {
        'Happy': "You look happy! I can see a positive expression on your face.",
        'Sad': "You seem a bit sad. Your expression shows some melancholy.",
        'Angry': "You appear upset or frustrated. Your expression looks tense.",
        'Fear': "You look concerned or worried about something.",
        'Surprise': "You look surprised! Your expression shows amazement.",
        'Neutral': "You have a calm, neutral expression. You look peaceful."
    }
    
    base_description = emotion_descriptions.get(emotion, f"Your expression appears to be {emotion.lower()}.")
    
    if confidence > 0.8:
        return base_description
    elif confidence > 0.5:
        return f"I think {base_description.lower()}"
    else:
        return "I can see your face, but I'm not completely certain about your expression right now."

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('voice_assistant.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('voice_assistant.html'), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, log_output=True)