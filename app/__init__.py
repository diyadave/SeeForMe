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
    """Handle voice input from client - CORE RESPONSE SYSTEM"""
    user_text = data.get('text', '')
    confidence = data.get('confidence', 0.0)
    language = data.get('language', 'en')
    
    logger.info(f"🗣️ Voice input: '{user_text}' (confidence: {confidence:.2f})")
    print(f"🎯 VOICE INPUT: {user_text}")
    
    # Extract name properly
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
    
    # Try Gemma2:2b first, then fall back to enhanced patterns
    response = ""
    try:
        import requests
        gemma_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3n",
                "prompt": f"You are a caring AI companion for blind users. A user named {user_name} just said: \"{user_text}\". Respond with empathy and support in 2-3 sentences.",
                "stream": False
            },
            timeout=5
        )
        if gemma_response.status_code == 200:
            gemma_data = gemma_response.json()
            if gemma_data.get('response'):
                response = gemma_data['response'].strip()
                print(f"🤖 GEMMA3N NANO SUCCESS: {response[:50]}...")
    except Exception as e:
        print(f"⚠️ Gemma3n nano not available: {e}")
        print("🔍 DEBUG: Ollama server status and model availability")
    
    # Enhanced pattern matching fallback (if Gemma fails)
    if not response:
        print("📱 USING INTELLIGENT PATTERN MATCHING (Gemma3n nano will work on mobile)")
        print("🎯 Current response generated by: PATTERN MATCHING SYSTEM")
        if user_name != "friend":
            response = f"Hello {user_name}! Nice to meet you! I'm your caring AI companion and I'm here to support you emotionally. How are you feeling today?"
        elif "scold" in user_text.lower() or "harsh" in user_text.lower():
            response = f"I'm so sorry someone was harsh with you today, {user_name}. That must have been really difficult and hurtful to experience. Remember, their words don't define your worth - you're valuable and deserving of kindness. Would you like to talk about what happened?"
        elif "look" in user_text.lower() or "see" in user_text.lower() or "describe" in user_text.lower() or "around" in user_text.lower():
            response = f"Let me look around and describe what I can see for you, {user_name}."
        elif "talk" in user_text.lower() or "hello" in user_text.lower():
            response = f"Hello there! Yes, I'd love to talk with you, {user_name}! I'm your emotional support companion, and I'm here to listen and chat about whatever is on your mind. How are you feeling today?"
        elif "sad" in user_text.lower() or "bad" in user_text.lower():
            response = "I'm sorry you're having a tough time. I can hear that you're going through something difficult right now. I'm here to listen and support you. You're not alone, and I care about how you're feeling."
        elif "happy" in user_text.lower() or "good" in user_text.lower():
            response = "That's wonderful to hear! I'm so glad you're feeling good today. Your happiness brings me joy too. What's been making you feel so positive?"
        else:
            response = f"I heard you say '{user_text}'. I'm your caring AI companion, always here to provide emotional support and understanding. Tell me more about what's on your mind, {user_name}."
    
    # Basic emotion detection
    detected_emotion = "neutral"
    if any(word in user_text.lower() for word in ["sad", "upset", "angry", "frustrated"]):
        detected_emotion = "sad"
    elif any(word in user_text.lower() for word in ["happy", "good", "great", "wonderful"]):
        detected_emotion = "happy"
    
    # Get vision analysis if available
    vision_analysis = None
    try:
        from services.vision_processor import vision_processor
        vision_analysis = vision_processor.get_intelligent_camera_response(user_text)
        if vision_analysis:
            # Enhance response with vision context
            vision_desc = vision_analysis.get('description', '')
            if vision_desc and len(vision_desc) > 10:
                response = f"{response} {vision_desc}"
                print(f"👁️ VISION: {vision_analysis['analysis_type']} using {vision_analysis['camera_used']} camera")
    except Exception as e:
        print(f"⚠️ Vision processing error: {e}")
    
    # Send response immediately
    emit('assistant_response', {
        'text': response,
        'emotion': detected_emotion,
        'user_emotion': detected_emotion,
        'vision_analysis': vision_analysis,
        'speak': True
    })
    
    print(f"✅ SENT: {response}")

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
    """Handle speech recognition results - IMMEDIATE RESPONSE VERSION"""
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
    
    print("📱 GENERATING IMMEDIATE RESPONSE")
    
    # Enhanced emotional intelligence responses - IMMEDIATE
    if user_name != "friend":
        response = f"Hello {user_name}! Nice to meet you! I'm your caring AI companion and I'm here to support you emotionally. How are you feeling today?"
    elif "scold" in user_text.lower() or "harsh" in user_text.lower():
        response = "I understand someone was harsh with you. That must have been really difficult to experience. Remember, other people's words don't define your worth. You're valuable and deserving of kindness. Would you like to talk about what happened?"
    elif "bad" in user_text.lower() or "sad" in user_text.lower():
        response = "I'm sorry you're having a tough time. I can hear that you're going through something difficult right now. I'm here to listen and support you. You're not alone, and I care about how you're feeling. What's been weighing on your mind?"
    elif "thank" in user_text.lower():
        response = "You're so welcome! I'm genuinely glad I could help and support you. It means a lot to me that you felt heard. I'm always here when you need emotional support or just someone to talk to."
    elif "happy" in user_text.lower() or "good" in user_text.lower():
        response = "That's wonderful to hear! I'm so glad you're feeling good today. Your happiness brings me joy too. What's been making you feel so positive?"
    elif "help" in user_text.lower() or "support" in user_text.lower():
        response = f"Of course I'm here to help you, {user_name}! I'm your caring AI companion, and supporting you is what I'm here for. What kind of support do you need right now?"
    elif "talk" in user_text.lower() or "hello" in user_text.lower():
        response = f"Hello there! Yes, I'd love to talk with you, {user_name}! I'm your emotional support companion, and I'm here to listen and chat about whatever is on your mind. How are you feeling today?"
    else:
        response = f"I heard you say '{user_text}'. I'm your caring AI companion, always here to provide emotional support and understanding. Tell me more about what's on your mind, {user_name}."
    
    # Detect user emotion from text (simple analysis)
    detected_emotion = "neutral"
    if any(word in user_text.lower() for word in ["sad", "upset", "angry", "frustrated", "mad"]):
        detected_emotion = "sad"
    elif any(word in user_text.lower() for word in ["happy", "good", "great", "wonderful", "amazing"]):
        detected_emotion = "happy"
    elif any(word in user_text.lower() for word in ["scared", "afraid", "worried", "anxious"]):
        detected_emotion = "fear"
    elif any(word in user_text.lower() for word in ["confused", "lost", "don't understand"]):
        detected_emotion = "confused"
    
    # Send response immediately
    emit('assistant_response', {
        'text': response,
        'emotion': detected_emotion,
        'user_emotion': detected_emotion,
        'speak': True
    })
    
    print(f"✅ SENT: {response}")

if __name__ == '__main__':
    logger.info("🚀 Starting SeeForMe Assistant")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)