#!/usr/bin/env python3
"""
Emotional AI Assistant Package Initialization
"""

from flask import Flask
from flask_socketio import SocketIO

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = '29524ddd0d294eaebcee5a09e00a7d37b84428f0a39beaf322d72845bd48e870'

# Initialize SocketIO with async_mode and CORS
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   logger=True,
                   engineio_logger=False)

# Import all modules after app is created to avoid circular imports
from app import speech_handler
from app import name_extractor
from app import gemma_connect
from app import tts_handler
from app import scene_detector
from app import camera_switcher