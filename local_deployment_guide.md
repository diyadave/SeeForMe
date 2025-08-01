# SeeForMe Local Deployment Guide

## Quick Start for Local Testing

### 1. Download and Extract
```bash
# Download the ZIP file and extract to your preferred location
cd /path/to/seeforMe
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install flask flask-socketio opencv-python face-recognition requests numpy pathlib

# Optional: Install additional packages for enhanced features
pip install vosk pyttsx3 gtts pygame
```

### 3. Set Up Your Gemma Model
```bash
# Make sure Ollama is running with your Gemma3n model
ollama serve
ollama run gemma3n:latest
```

### 4. Configure DroidCam for Camera Testing

**Option A: DroidCam Setup**
1. Install DroidCam on your phone and PC
2. Connect phone via USB or WiFi
3. DroidCam creates a virtual camera at `/dev/video2` (Linux) or similar

**Option B: Update Camera Configuration**
Edit `services/vision_processor.py` to use DroidCam:
```python
# Find the get_current_frame method and update camera index
cap = cv2.VideoCapture(2)  # DroidCam usually appears as index 2
```

### 5. Run SeeForMe
```bash
# Start the complete offline assistant
python main.py

# Or use Flask development server
flask run --host=0.0.0.0 --port=5000
```

### 6. Access the App
- Open browser: http://localhost:5000
- Grant microphone permissions
- Grant camera permissions
- Start testing voice commands

## Testing All Features

### Memory System Test
1. Say: "My name is [YourName]"
2. Say: "I am feeling sad today"
3. Close and reopen app
4. Say: "Hello" 
   - Expected: App mentions yesterday's sadness

### Face Recognition Test
1. Position phone camera to see a person
2. Say: "What do you see around me?" (triggers back camera)
3. Have person say: "My name is [PersonName]"
4. Move camera away and back
5. Say: "Describe the room"
   - Expected: App recognizes the person by name

### Camera Switching Test
1. Say: "Hello" (should use front camera for emotion)
2. Say: "How do I look?" (front camera)
3. Say: "What do you see around me?" (switches to back camera)
4. Say: "Describe the room" (back camera for scene)

### Gemma Integration Test
- Your local Gemma3n model should provide much better responses
- Responses will include conversation context
- Emotional support will be more sophisticated

## File Structure After Download
```
seeforMe/
├── main.py                 # Main application entry
├── app/
│   └── __init__.py        # Flask app with SocketIO
├── services/
│   ├── memory_manager.py   # Offline JSON memory
│   ├── face_memory.py     # Face recognition
│   ├── vision_processor.py # Camera processing
│   └── [other services]
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── [CSS/JS files]
├── memory_data/           # Created automatically
│   ├── conversations.json
│   ├── emotional_states.json
│   ├── user_profiles.json
│   └── known_faces.json
└── models/                # AI model files (optional)
```

## Expected Voice Commands

**Emotional Interaction:**
- "Hello" / "Hi there" 
- "I am feeling sad/happy/worried"
- "Someone scolded me today"
- "I had a great day"

**Environment Awareness:**
- "What do you see around me?"
- "Describe the room"
- "Who is here?"
- "Look around"

**Social Introduction:**
- "My name is [Name]"
- Have others say: "My name is [Name]"

**Memory Testing:**
- After closing/reopening: "Hello"
- "How am I feeling?"
- "Do you remember me?"

## Troubleshooting

**Camera Issues:**
- Check DroidCam connection
- Update camera index in vision_processor.py
- Verify camera permissions in browser

**Gemma Connection:**
- Ensure Ollama is running: `ollama serve`
- Check model is loaded: `ollama run gemma3n`
- Verify localhost:11434 is accessible

**Memory Files:**
- Check `memory_data/` folder is created
- JSON files should appear after first interactions
- Files grow with usage

**Audio Issues:**
- Grant microphone permissions
- Check browser audio settings
- Test with simple "Hello" command

Your SeeForMe app is now ready for comprehensive local testing with full camera functionality, face recognition, emotional memory, and your powerful Gemma3n model!