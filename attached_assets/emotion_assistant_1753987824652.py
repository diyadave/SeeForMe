import cv2
import numpy as np
import onnxruntime as ort
import pyttsx3
import requests
import time
import threading
import queue
import random
import json
import pygame
from collections import deque
import pyaudio
import os
import webrtcvad
import collections
import sys
from vosk import Model, KaldiRecognizer
import re

# Initialize pygame mixer for audio
pygame.mixer.init()

# System states for seamless flow
class SystemState:
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    EMOTION_ONLY = "emotion_only"
    SPEAKING = "speaking"

class OfflineEmotionalAI:
    """Fully automatic offline emotional AI assistant with VAD"""
    
    def __init__(self):
        self.current_state = SystemState.EMOTION_ONLY
        
        # Core components
        self.emotion_session = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.tts_engine = None
        self.vosk_recognizer = None
        
        # VAD and audio components
        self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # State management
        self.is_speaking = False
        self.conversation_buffer = collections.deque(maxlen=100)
        self.speech_frames = collections.deque(maxlen=50)
        self.silence_threshold = 30  # frames of silence to end speech
        self.speech_threshold = 10   # frames of speech to start processing
        self.is_recording = False  
        self.last_voice_time = time.time()  
        
        # Emotion detection
        self.last_emotion = "Neutral"
        self.emotion_history = deque(maxlen=5)
        self.last_emotional_response_time = 0
        self.emotion_cooldown = 15  # seconds
        
        # Conversation context
        self.user_name = None
        self.conversation_history = []
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.response_queue = queue.Queue(maxsize=5)
        self.shutdown_event = threading.Event()
        
        # Gemma availability
        self.gemma_available = False
        
        # Initialize all components
        self.init_components()
        
    def init_components(self):
        """Initialize all AI components"""
        print("🚀 Initializing Offline Emotional AI Assistant...")
        
        # Load emotion detection model
        try:
            self.emotion_session = ort.InferenceSession("emotion_model.onnx")
            print("✅ Emotion detection model loaded")
        except Exception as e:
            print(f"❌ Failed to load emotion model: {e}")
            sys.exit(1)
        
        # Initialize TTS
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 160)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Select best voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voices for emotional support
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['female', 'zira', 'eva']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            print("✅ Text-to-Speech initialized")
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            sys.exit(1)

        # Vosk initialization
        model_path = "models/vosk-model-small-en-us-0.15"
        if os.path.exists(model_path):
            self.vosk_model = Model(model_path)
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
            self.vosk_recognizer.SetWords(True)
            print("✅ Offline speech recognition initialized")
        else:
            print(f"❌ Vosk model not found at {model_path}")
            sys.exit(1)

        # Test Gemma connection
        self.test_gemma_connection()
        
        # Initialize audio system
        self.init_audio_system()
        
        # Start background threads
        self.start_background_threads()
        
        print("🎯 All systems ready! Starting automatic emotional AI...")
    
    def test_gemma_connection(self):
        """Test and verify Gemma 3n connection with retry"""
        print("🔍 Testing Gemma 3n connection...")
        while not self.gemma_available:
            try:
                response = requests.post(
                    "http://127.0.0.1:11434/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "gemma3n",
                        "messages": [
                            {"role": "user", "content": "Hello"}
                        ],
                        "stream": False
                    },
                    timeout=120
                )
                if response.ok:
                    result = response.json()
                    if "message" in result and "content" in result["message"]:
                        self.gemma_available = True
                        print("✅ Gemma 3n connection verified and working")
                        return
                print("⏳ Gemma responded but incomplete. Retrying in 20 seconds...")
            except Exception as e:
                print(f"⏳ Waiting for Gemma to wake up... retrying in 20 seconds ({e})")
                time.sleep(20)
            
    def init_audio_system(self):
        """Initialize VAD-based audio system"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find best microphone
            device_id = self.find_best_microphone()
            if device_id is None:
                print("❌ No suitable microphone found")
                sys.exit(1)
            
            # Start audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.frame_size
            )
            
            print("✅ Audio system initialized with VAD")
            
        except Exception as e:
            print(f"❌ Audio system initialization failed: {e}")
            sys.exit(1)
    
    def find_best_microphone(self):
        """Find the best available microphone"""
        print("🎤 Scanning for optimal microphone...")
        
        best_device = None
        for i in range(self.pyaudio_instance.get_device_count()):
            try:
                info = self.pyaudio_instance.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    # Test the device
                    test_stream = self.pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=self.frame_size
                    )
                    
                    # Quick test
                    data = test_stream.read(self.frame_size)
                    test_stream.close()
                    
                    if data:
                        best_device = i
                        print(f"✅ Selected microphone: {info['name']}")
                        break
                        
            except Exception:
                continue
        
        return best_device
    
    def start_background_threads(self):
        """Start all background processing threads"""
        
        # Start improved audio processing with silence handling
        self.audio_thread = threading.Thread(target=self.start_audio_processing, daemon=True)
        self.audio_thread.start()
        
        # Start response generation thread
        self.response_thread = threading.Thread(target=self.response_processing_loop, daemon=True)
        self.response_thread.start()
        
        print("✅ Background threads started")

    def start_audio_processing(self):
        """Continuously listen for speech and detect voice activity"""
        print("🎤 Audio processing started - listening for speech...")

        audio_data = b''
        silence_counter = 0

        while not self.shutdown_event.is_set():
            try:
                data = self.audio_stream.read(self.frame_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                if self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate):
                    silence_counter = 0
                    self.last_voice_time = time.time()
                    audio_data += data

                    if not self.is_recording:
                        self.is_recording = True
                        print("🎤 Speech detected - recording...")

                else:
                    if self.is_recording:
                        silence_counter += 1
                        if silence_counter > 30 and time.time() - self.last_voice_time > 3:
                            print("🎤 End of speech detected - processing...")
                            self.process_speech(audio_data)
                            audio_data = b''
                            silence_counter = 0
                            self.is_recording = False

            except Exception as e:
                print(f"❌ Audio read error: {e}")
                continue
    
    def process_speech(self, audio_data):
        """Process captured speech with Vosk"""
        self.current_state = SystemState.PROCESSING
        
        try:
            # Reset recognizer for clean processing
            self.vosk_recognizer = KaldiRecognizer(
                self.vosk_model, self.sample_rate
            )
            self.vosk_recognizer.SetWords(True)
            
            # Process audio in chunks
            chunk_size = self.frame_size * 4
            results = []
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) >= self.frame_size:
                    if self.vosk_recognizer.AcceptWaveform(chunk):
                        result = json.loads(self.vosk_recognizer.Result())
                        if result.get('text'):
                            results.append(result['text'])
            
            # Get final result
            final_result = json.loads(self.vosk_recognizer.FinalResult())
            if final_result.get('text'):
                results.append(final_result['text'])
            
            # Combine results
            user_text = ' '.join(results).strip()
            
            if user_text:
                print(f"👤 User said: '{user_text}'")
                
                # Extract name if mentioned
                name = self.extract_name(user_text)
                if name:
                    self.user_name = name
                    print(f"👤 User name saved: {self.user_name}")
                
                # Add to conversation history
                self.conversation_history.append({"role": "user", "text": user_text, "emotion": self.last_emotion})
                
                # Generate response immediately using Gemma
                self.generate_and_speak_response(user_text, self.last_emotion)
            else:
                print("❌ No clear speech recognized")
                self.current_state = SystemState.EMOTION_ONLY
                
        except Exception as e:
            print(f"❌ Speech processing error: {e}")
            self.current_state = SystemState.EMOTION_ONLY
    
    def response_processing_loop(self):
        """Background thread for processing queued responses"""
        while not self.shutdown_event.is_set():
            try:
                item = self.response_queue.get(timeout=1)
                if item is None:
                    break
                
                response_text = item
                self.speak_response(response_text)
                
                self.response_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Response processing error: {e}")
    
    def extract_name(self, user_text):
        """Stricter name extraction to avoid saving wrong names from noisy STT transcription."""
        # Start of utterance only, ignore obviously wrong names
        patterns = [
            r"^\s*my name is ([a-zA-Z]{3,})\b",
            r"^\s*call me ([a-zA-Z]{3,})\b",
            r"\bi am ([a-zA-Z]{3,})\b",
            r"\bi'm ([a-zA-Z]{3,})\b",
            r"\bthis is ([a-zA-Z]{3,})\b"
        ]
        stoplist = {'not', 'yeah', 'feeling', 'please', 'can', 'the', 'is', 'more', 'at', 'should', 'no', 'yes'}
        for pattern in patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                possible_name = match.group(1).capitalize()
                if possible_name.lower() not in stoplist:
                    # Only set if user_name not set, or to change ask for confirmation
                    if not self.user_name:
                        return possible_name
                    elif possible_name != self.user_name:
                        print(f"🔍 Detected possible new name '{possible_name}', won't overwrite '{self.user_name}' without confirmation.")
        return None

    def generate_and_speak_response(self, user_text, current_emotion):
        """Generate response using Gemma and speak it immediately"""
        self.current_state = SystemState.RESPONDING
        
        if self.gemma_available:
            response = self.get_gemma_response(user_text, current_emotion)
        else:
            response = self.get_fallback_response(user_text, current_emotion)
        
        if response:
            # Clean up the response
            response = self.clean_response(response)
            print(f"🤖 Generated response: {response}")
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "text": response})
            
            # Speak immediately
            self.speak_response(response)
        else:
            print("❌ No response generated")
            self.current_state = SystemState.EMOTION_ONLY

    def get_gemma_response(self, user_text, emotion):
        try:
            context_prompt = f"User's emotion: {emotion}\n"
            if self.user_name:
                context_prompt += f"User's name: {self.user_name}\n"

            if len(self.conversation_history) > 0:
                context_prompt += "Recent conversation:\n"
                recent = self.conversation_history[-4:]
                for entry in recent:
                    role = "User" if entry["role"] == "user" else "Assistant"
                    context_prompt += f"{role}: {entry['text']}\n"

            context_prompt += f"\nUser just said: \"{user_text}\"\n"
            context_prompt += "Respond as a warm, caring friend providing emotional support. Keep response to 1-2 sentences."

            payload = {
                "model": "gemma3n",
                "messages": [
                    {"role": "user", "content": context_prompt}
                ],
                "stream": False
            }

            response = requests.post(
                "http://127.0.0.1:11434/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )

            if response.ok:
                result = response.json()
                generated_text = result.get("message", {}).get("content", "").strip()
                if generated_text:
                    print(f"✅ Gemma responded: {generated_text}")
            if response.ok:
                result = response.json()
                generated_text = result.get("message", {}).get("content", "").strip()
                if generated_text:
                    print(f"✅ Gemma responded: {generated_text}")
                    return generated_text
                else:
                    print("❌ Gemma returned empty response. Retrying once...")
                    time.sleep(5)
                    return self.get_fallback_response(user_text, emotion)
            else:
                print(f"❌ Gemma API error: {response.status_code} {response.text}")
                return self.get_fallback_response(user_text, emotion)

        except requests.exceptions.Timeout:
            print("❌ Gemma request timed out. Retrying once after delay...")
            time.sleep(5)
            return self.get_fallback_response(user_text, emotion)
        except Exception as e:
            print(f"❌ Gemma error: {e}")
            return self.get_fallback_response(user_text, emotion)
        if len(self.conversation_history) > 0:
            context += "\n\nRecent conversation:"
            # Include last 3 exchanges for context
            recent = self.conversation_history[-6:]  # Last 3 user-assistant pairs
            for entry in recent:
                role = "User" if entry["role"] == "user" else "You"
                context += f"\n{role}: {entry['text']}"
        
        return context

    def clean_response(self, response):
        """Clean up the generated response"""
        # Remove common unwanted prefixes/suffixes
        unwanted_patterns = [
            r"^(Assistant|AI|Bot):\s*",
            r"^Response:\s*",
            r"^\*.*?\*\s*",
            r"\n.*$",  # Remove anything after newline
        ]
        
        cleaned = response.strip()
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Ensure it ends with proper punctuation
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += "."
        
        return cleaned.strip()

    def get_fallback_response(self, user_text, emotion):
        """Generate fallback responses when Gemma is unavailable"""
        text_lower = user_text.lower()

        # Handle name introductions
        if self.user_name and any(phrase in text_lower for phrase in ['my name is', 'i am', 'i\'m', 'call me']):
            return f"It's wonderful to meet you, {self.user_name}! I'm so glad you're here."

        # Handle greetings
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            if self.user_name:
                return f"Hello {self.user_name}! How are you feeling today?"
            else:
                return "Hello there! It's great to connect with you. How are you feeling?"

        # Emotion-based responses
        if emotion == "Happy":
            return random.choice([
                "I can hear the joy in your voice! What's making you so happy?",
                "Your happiness is absolutely contagious! I love that energy.",
                "That's wonderful to hear! Tell me more about what's bringing you such joy."
            ])
        elif emotion == "Sad":
            return random.choice([
                "I can hear that you're going through something difficult. I'm here for you.",
                "It sounds like you're carrying some heavy feelings right now. You don't have to face them alone.",
                "Thank you for sharing that with me. Your feelings are completely valid."
            ])
        elif emotion == "Angry":
            return random.choice([
                "I can hear the frustration in your voice. Sometimes it helps to talk through these intense feelings.",
                "You sound really upset about something. I'm here to listen without judgment.",
                "Those are some strong emotions you're experiencing. Take a deep breath with me."
            ])
        elif emotion == "Fear":
            return random.choice([
                "I can hear the worry in your voice. You're safe here, and we can work through this together.",
                "It sounds like something is making you anxious. You're braver than you know.",
                "I sense you're feeling uneasy. Remember, you have the strength to handle whatever comes your way."
            ])
        else:
            # General supportive responses
            return random.choice([
                "Thank you for sharing that with me. I really appreciate your openness.",
                "I'm here and listening. Tell me more about what's on your mind.",
                "Your thoughts and feelings matter to me. I'm glad you felt comfortable sharing."
            ])

    def speak_response(self, text):
        """Speak the response using TTS"""
        self.current_state = SystemState.SPEAKING
        self.is_speaking = True
        
        try:
            print(f"🔊 Speaking: {text}")
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            print("✅ Response delivered")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
        
        finally:
            self.is_speaking = False
            self.current_state = SystemState.EMOTION_ONLY
    
    def process_emotion_detection(self, frame):
        """Process emotion detection from camera frame"""
        if self.current_state in [SystemState.SPEAKING, SystemState.PROCESSING]:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        current_time = time.time()
        
        for (x, y, w, h) in faces:
            try:
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_processed = cv2.resize(face_roi, (48, 48)).astype(np.float32) / 255.0
                face_input = face_processed.reshape(1, 48, 48, 1)
                
                # Predict emotion
                input_name = self.emotion_session.get_inputs()[0].name
                outputs = self.emotion_session.run(None, {input_name: face_input})
                prediction = np.argmax(outputs[0])
                emotion = self.emotion_labels[prediction]
                confidence = np.max(outputs[0])
                
                # Update emotion history
                self.emotion_history.append(emotion)
                
                # Get stable emotion
                if len(self.emotion_history) >= 3:
                    stable_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                    
                    # Draw emotion on frame
                    color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{emotion} ({confidence:.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Update current emotion
                    if (stable_emotion != self.last_emotion and 
                        confidence > 0.65 and
                        current_time - self.last_emotional_response_time > self.emotion_cooldown and
                        stable_emotion not in ['Neutral']):
                        
                        self.last_emotion = stable_emotion
                        self.last_emotional_response_time = current_time
                        
                        # Generate appropriate emotional response only if not in conversation
                        if self.current_state == SystemState.EMOTION_ONLY:
                            emotion_response = self.get_emotion_only_response(stable_emotion)
                            if emotion_response:
                                threading.Thread(
                                    target=self.speak_response, 
                                    args=(emotion_response,), 
                                    daemon=True
                                ).start()
                    
                    self.last_emotion = stable_emotion
                    
            except Exception as e:
                print(f"❌ Emotion detection error: {e}")
        
        return frame
    
    def get_emotion_only_response(self, emotion):
        """Get response for emotion-only detection (no speech)"""
        if self.gemma_available:
            try:
                prompt = f"The user is showing {emotion} emotion on their face but hasn't spoken."
                if self.user_name:
                    prompt += f" The user's name is {self.user_name}."
                prompt += " Respond as a caring friend who notices their emotional state. Keep it brief and supportive (1 sentence)."

                response = requests.post(
                    "http://127.0.0.1:11434/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "gemma3n",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False
                    },
                    timeout=120
                )

                if response.ok:
                    result = response.json()
                    generated = result.get("message", {}).get("content", "").strip()
                    if generated:
                        return self.clean_response(generated)
                        
            except Exception as e:
                print(f"❌ Gemma emotion response error: {e}")
    
        # Fallback responses for emotion-only detection
        responses = {
            'Happy': [
                f"I can see that beautiful smile{f', {self.user_name}' if self.user_name else ''}! Your happiness is wonderful to see.",
                f"You look so joyful right now{f', {self.user_name}' if self.user_name else ''}! What's bringing you such wonderful energy?",
                f"That radiant expression makes me so happy to see{f', {self.user_name}' if self.user_name else ''}."
            ],
            'Sad': [
                f"I can see the sadness in your eyes{f', {self.user_name}' if self.user_name else ''}. I'm here for you.",
                f"You look like you're carrying something heavy{f', {self.user_name}' if self.user_name else ''}. I'm here to listen.",
                f"I notice you seem down{f', {self.user_name}' if self.user_name else ''}. You don't have to face this alone."
            ],
            'Angry': [
                f"I can see you're feeling frustrated{f', {self.user_name}' if self.user_name else ''}. Take a deep breath with me.",
                f"You look upset about something{f', {self.user_name}' if self.user_name else ''}. Sometimes talking helps.",
                f"I notice those intense emotions{f', {self.user_name}' if self.user_name else ''}. I'm here if you need support."
            ],
            'Fear': [
                f"I can see worry in your expression{f', {self.user_name}' if self.user_name else ''}. You're safe here with me.",
                f"You look anxious{f', {self.user_name}' if self.user_name else ''}, and that's completely understandable.",
                f"I sense you're feeling uneasy{f', {self.user_name}' if self.user_name else ''}. You're stronger than you know."
            ],
            'Surprise': [
                f"You look amazed by something{f', {self.user_name}' if self.user_name else ''}! I love seeing that expression of wonder.",
                f"Something caught you off guard{f', {self.user_name}' if self.user_name else ''}! Life has such interesting surprises.",
                f"That look of astonishment is precious{f', {self.user_name}' if self.user_name else ''}."
            ],
            'Disgust': [
                f"I can see something is bothering you{f', {self.user_name}' if self.user_name else ''}. Your feelings are valid.",
                f"You seem troubled by something{f', {self.user_name}' if self.user_name else ''}. Trust your instincts.",
                f"I notice you're uncomfortable{f', {self.user_name}' if self.user_name else ''}. That's okay."
            ]
        }
        
        if emotion in responses:
            return random.choice(responses[emotion])
        return None

    def run(self):
        """Main execution loop"""
        # Initialize camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("\n🎯 Automatic Emotional AI Assistant Active!")
        print(f"🤖 Gemma 3n: {'✅ Connected' if self.gemma_available else '❌ Using fallbacks'}")
        print("🎤 Continuously listening for speech...")
        print("👁️ Monitoring facial emotions...")
        print("💬 No buttons needed - just speak naturally!")
        print("\nPress 'q' to quit, 'h' for help, 'g' to test Gemma")
        
        frame_skip = 2
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera read failed")
                    break
                
                frame_count += 1
                
                # Process emotion detection on some frames
                if frame_count % frame_skip == 0:
                    frame = self.process_emotion_detection(frame)
                
                # Add status overlay
                status_color = {
                    SystemState.EMOTION_ONLY: (0, 255, 0),    # Green
                    SystemState.LISTENING: (0, 255, 255),     # Yellow
                    SystemState.PROCESSING: (255, 0, 255),    # Magenta
                    SystemState.RESPONDING: (0, 0, 255),      # Red
                    SystemState.SPEAKING: (0, 165, 255)       # Orange
                }
                
                cv2.putText(frame, f"Status: {self.current_state}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color.get(self.current_state, (255, 255, 255)), 2)
                
                cv2.putText(frame, f"Emotion: {self.last_emotion}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if self.user_name:
                    cv2.putText(frame, f"User: {self.user_name}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Emotional AI Assistant', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Shutting down...")
                    break
                elif key == ord('h'):
                    self.show_help()
                elif key == ord('g'):
                    self.test_gemma_connection()
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        
        finally:
            # Cleanup
            self.shutdown_event.set()
            cap.release()
            cv2.destroyAllWindows()
            
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            print("✅ All resources released. Goodbye!")

    def show_help(self):
        """Display help information"""
        help_text = [
            "\n🌟 Emotional AI Assistant Help 🌟",
            "--------------------------------",
            "This is a fully automatic offline emotional AI assistant that:",
            "- Continuously listens for your voice (no button press needed)",
            "- Detects your facial emotions in real-time",
            "- Responds with empathy and understanding",
            "- Works completely offline (except optional Gemma 3n connection)",
            "",
            "🔹 Just speak naturally to start a conversation",
            "🔹 The AI will notice your emotions even if you don't speak",
            "🔹 Say things like 'My name is...' to personalize the experience",
            "",
            "Keyboard Controls:",
            "  q - Quit the application",
            "  h - Show this help message",
            "  g - Test Gemma 3n connection",
            "",
            "Current Status:",
            f"- Gemma 3n: {'✅ Connected' if self.gemma_available else '❌ Not available'}",
            f"- User Name: {self.user_name if self.user_name else 'Not set'}",
            f"- Current Emotion: {self.last_emotion}",
            "",
            "Note: For best results, ensure:",
            "- Good lighting for emotion detection",
            "- Clear speech without background noise",
            "- Camera is positioned at face level"
        ]
        
        print("\n".join(help_text))

    def shutdown(self):
        """Clean shutdown of the system"""
        print("\n🛑 Beginning graceful shutdown...")
        self.shutdown_event.set()
        
        # Stop audio processing
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        # Stop TTS if speaking
        if self.tts_engine._inLoop:
            self.tts_engine.endLoop()
        
        # Join threads
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1)
        if hasattr(self, 'response_thread'):
            self.response_thread.join(timeout=1)
        
        print("✅ System shutdown complete")


if __name__ == "__main__":
    try:
        ai = OfflineEmotionalAI()
        ai.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        if 'ai' in locals():
            ai.shutdown()
        sys.exit(0)