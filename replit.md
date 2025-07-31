# Offline Smart Assistant - SeeForMe

## Overview

SeeForMe is a fully offline smart assistant application designed specifically for visually impaired users. The application combines voice interaction, emotion detection, scene analysis, and natural conversation to provide emotional support and environmental awareness. Built with Python Flask and real-time communication via SocketIO.

**Current Status (Jan 31, 2025):**
- ✅ **Working**: Vosk speech recognition, Pattern-based emotional responses, TTS output, Web interface
- 🔄 **In Progress**: Gemma3n integration (Ollama server setup needed)
- ⏳ **Planned**: YOLOv8n computer vision, Emotion detection ONNX models, Scene analysis

The app currently provides immediate emotional support through pattern matching while the full AI pipeline is being integrated.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular, service-oriented architecture with the following key components:

### Backend Architecture
- **Flask Web Framework**: Serves the web interface and handles HTTP requests
- **SocketIO**: Manages real-time bidirectional communication between frontend and backend
- **Threading-based Processing**: All AI components run in separate threads to prevent blocking
- **Service Layer Pattern**: Modular services for different AI capabilities

### Frontend Architecture
- **Web-based Interface**: HTML/CSS/JavaScript with Bootstrap for responsive design
- **Real-time Updates**: JavaScript client communicates with backend via SocketIO
- **Accessibility-first Design**: High contrast, screen reader friendly, keyboard navigation

## Key Components

### 1. Assistant Coordinator (`services/assistant_coordinator.py`)
Central orchestration service that manages all other components and maintains application state.

**Purpose**: Coordinates communication between speech recognition, AI processing, camera management, and TTS
**Key Features**:
- Thread-safe component management
- User context tracking (name, emotion, conversation history)
- Camera mode switching (scene vs emotion detection)
- State management and persistence

### 2. Speech Recognition (`services/speech_handler.py`)
Offline speech recognition using Vosk models for English, Hindi, and Gujarati.

**Purpose**: Convert user speech to text for processing
**Key Features**:
- Multi-language support with auto-detection
- Continuous listening with voice activity detection
- Configurable audio parameters
- Fallback handling for missing dependencies

### 3. Natural Language Processing (`services/gemma_connect.py`)
Integration with Gemma 3n language model via Ollama for conversational AI.

**Purpose**: Generate empathetic, contextual responses to user input
**Key Features**:
- Local Ollama server communication
- Connection pooling and retry logic
- Performance monitoring and caching
- Context-aware prompt engineering for accessibility use cases

### 4. Scene Detection (`services/scene_detector.py`)
Computer vision pipeline combining multiple AI models for environmental understanding.

**Purpose**: Analyze camera feed to describe surroundings and detect people/emotions
**Key Features**:
- YOLOv8n object detection
- Places365 scene classification
- Facial emotion recognition via ONNX models
- Real-time frame processing

### 5. Text-to-Speech (`services/tts_handler.py`)
Multi-language speech synthesis for assistant responses.

**Purpose**: Convert AI-generated text responses to speech
**Key Features**:
- Offline English TTS via pyttsx3
- Online fallback for Hindi/Gujarati via gTTS
- Audio caching and playback management
- Voice customization options

### 6. Name Extraction (`services/name_extractor.py`)
Natural language processing for extracting user names from speech.

**Purpose**: Personalize interactions by learning and using user names
**Key Features**:
- Multi-language name extraction patterns
- Stopword filtering to avoid false positives
- Context-aware name validation

## Data Flow

1. **Audio Input**: User speaks → Vosk converts to text
2. **Intent Analysis**: Text analyzed for emotional context and intent
3. **Camera Switching**: Based on intent, switches between front (emotion) and back (scene) camera
4. **AI Processing**: 
   - Scene: YOLOv8n + Places365 analyze environment
   - Emotion: ONNX model detects user's facial emotion
5. **Context Building**: Results combined with user history and emotional state
6. **Response Generation**: Gemma 3n generates empathetic response using full context
7. **Audio Output**: pyttsx3 converts response to speech
8. **UI Updates**: SocketIO sends real-time status updates to frontend

## External Dependencies

### AI Models (Downloaded locally)
- **Vosk Models**: Speech recognition for multiple languages
- **YOLOv8n**: Object detection model
- **Emotion ONNX Model**: Facial emotion recognition
- **Places365**: Scene classification
- **Gemma 3n**: Language model via Ollama

### Python Libraries
- **Core**: Flask, SocketIO, threading, queue
- **Audio**: pyaudio, pyttsx3, pygame
- **Computer Vision**: OpenCV, ultralytics, torch, onnxruntime
- **ML/AI**: vosk, requests (for Ollama API)
- **Utilities**: pathlib, json, logging

### Hardware Requirements
- **Microphone**: For speech input
- **Camera(s)**: Front camera for emotion detection, back camera for scene analysis
- **Speakers/Headphones**: For TTS output

## Deployment Strategy

### Local Development
- All AI models run locally for complete offline functionality
- Ollama server hosts Gemma 3n model locally
- No internet connection required after initial setup

### Model Setup Process
1. Install Ollama and download Gemma 3n model
2. Download Vosk language models for speech recognition
3. Download YOLOv8n weights
4. Download emotion detection ONNX model
5. Set up Places365 model files

### Performance Considerations
- **Threading**: All AI processing runs in separate threads
- **Memory Management**: Models loaded once and cached
- **Real-time Processing**: Camera feed processed at optimized intervals
- **Fallback Systems**: Graceful degradation when models unavailable

### Accessibility Compliance
- **WCAG 2.1 AA**: High contrast design, keyboard navigation
- **Screen Reader Support**: Semantic HTML, ARIA labels
- **Voice-first Interface**: Primary interaction through speech
- **Multi-language**: Support for English, Hindi, Gujarati

The application is designed to work completely offline once properly configured, ensuring privacy and reliability for users who need consistent access to assistive technology.