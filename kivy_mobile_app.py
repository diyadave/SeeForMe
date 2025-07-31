"""
SeeForMe Mobile App - Kivy Implementation
Fully offline smart assistant for blind users
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.logger import Logger
import threading
import subprocess
import time
import requests

# Import our Flask backend
from app import app as flask_app, socketio

class SeeForMeApp(App):
    def __init__(self):
        super().__init__()
        self.flask_server = None
        self.server_thread = None
        self.is_listening = False
        
    def build(self):
        """Build the main UI"""
        Logger.info("SeeForMe: Building mobile interface")
        
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Title
        title = Label(
            text='SeeForMe - Voice Assistant',
            font_size='24sp',
            size_hint_y=0.2,
            color=(1, 1, 1, 1)
        )
        
        # Status label
        self.status_label = Label(
            text='Starting...',
            font_size='18sp',
            size_hint_y=0.3,
            color=(0.8, 0.8, 0.8, 1)
        )
        
        # Response area
        self.response_label = Label(
            text='Say "Hello" to begin conversation',
            font_size='16sp',
            size_hint_y=0.4,
            text_size=(None, None),
            halign='center',
            valign='middle',
            color=(0.9, 0.9, 0.9, 1)
        )
        
        # Control button
        self.control_button = Button(
            text='Start Listening',
            font_size='20sp',
            size_hint_y=0.1,
            background_color=(0, 0.7, 0, 1)
        )
        self.control_button.bind(on_press=self.toggle_listening)
        
        # Add all widgets
        layout.add_widget(title)
        layout.add_widget(self.status_label)
        layout.add_widget(self.response_label)
        layout.add_widget(self.control_button)
        
        # Start Flask server in background
        Clock.schedule_once(self.start_backend, 1)
        
        return layout
    
    def start_backend(self, dt):
        """Start Flask backend server"""
        Logger.info("SeeForMe: Starting backend server")
        self.status_label.text = "Starting AI backend..."
        
        # Start Flask server in separate thread
        self.server_thread = threading.Thread(target=self.run_flask_server, daemon=True)
        self.server_thread.start()
        
        # Wait a bit, then test connection
        Clock.schedule_once(self.test_backend_connection, 3)
    
    def run_flask_server(self):
        """Run Flask server"""
        try:
            Logger.info("SeeForMe: Flask server starting on port 5000")
            socketio.run(flask_app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)
        except Exception as e:
            Logger.error(f"SeeForMe: Flask server error: {e}")
    
    def test_backend_connection(self, dt):
        """Test if backend is running"""
        try:
            response = requests.get('http://127.0.0.1:5000/', timeout=5)
            if response.status_code == 200:
                Logger.info("SeeForMe: Backend connected successfully")
                self.status_label.text = "AI Backend Ready"
                self.control_button.background_color = (0, 0.8, 0, 1)
                self.control_button.text = "Start Voice Assistant"
            else:
                self.status_label.text = "Backend connection failed"
        except Exception as e:
            Logger.error(f"SeeForMe: Backend test failed: {e}")
            self.status_label.text = "Backend starting..."
            # Retry in 2 seconds
            Clock.schedule_once(self.test_backend_connection, 2)
    
    def toggle_listening(self, instance):
        """Toggle voice listening"""
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """Start voice recognition"""
        Logger.info("SeeForMe: Starting voice recognition")
        self.is_listening = True
        self.control_button.text = "Stop Listening"
        self.control_button.background_color = (0.8, 0, 0, 1)
        self.status_label.text = "Listening for your voice..."
        
        # Here you would integrate with Android/iOS speech recognition
        # For now, simulate with text input
        self.simulate_voice_input()
    
    def stop_listening(self):
        """Stop voice recognition"""
        Logger.info("SeeForMe: Stopping voice recognition")
        self.is_listening = False
        self.control_button.text = "Start Voice Assistant"
        self.control_button.background_color = (0, 0.8, 0, 1)
        self.status_label.text = "AI Backend Ready"
    
    def simulate_voice_input(self):
        """Simulate voice input for testing"""
        # This would be replaced with actual speech recognition
        test_phrases = [
            "Hello my name is Diya",
            "I am feeling sad today",
            "Someone scold me today",
            "I want to talk with you"
        ]
        
        def send_test_input():
            for phrase in test_phrases:
                if not self.is_listening:
                    break
                    
                Logger.info(f"SeeForMe: Simulating input: {phrase}")
                self.process_voice_input(phrase)
                time.sleep(5)
        
        # Run simulation in background
        threading.Thread(target=send_test_input, daemon=True).start()
    
    def process_voice_input(self, text):
        """Process voice input and get AI response"""
        try:
            # Send to Flask backend
            response = requests.post(
                'http://127.0.0.1:5000/api/voice_input',
                json={'text': text, 'confidence': 1.0, 'language': 'en'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get('text', 'No response')
                
                # Update UI
                Clock.schedule_once(lambda dt: self.update_response(ai_response), 0)
                
        except Exception as e:
            Logger.error(f"SeeForMe: Voice processing error: {e}")
            Clock.schedule_once(lambda dt: self.update_response("Sorry, I couldn't process that."), 0)
    
    def update_response(self, response_text):
        """Update the response display"""
        self.response_label.text = response_text
        Logger.info(f"SeeForMe: AI Response: {response_text}")
        
        # Here you would trigger text-to-speech
        self.speak_response(response_text)
    
    def speak_response(self, text):
        """Convert text to speech"""
        # This would integrate with Android/iOS TTS
        Logger.info(f"SeeForMe: Speaking: {text[:50]}...")
        
        # For desktop testing, use system TTS
        try:
            if hasattr(self, 'tts_process'):
                self.tts_process.terminate()
            
            # Use espeak on Linux (install with: apt install espeak)
            self.tts_process = subprocess.Popen([
                'espeak', '-s', '150', '-v', 'en', text
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            Logger.error(f"SeeForMe: TTS error: {e}")


# Add API endpoint for mobile communication
@flask_app.route('/api/voice_input', methods=['POST'])
def api_voice_input():
    """API endpoint for voice input from mobile app"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Process through existing voice_input handler logic
        # (This would call the same logic as the SocketIO handler)
        
        # For now, return a simple response
        return {
            'text': f"I heard you say: {text}",
            'emotion': 'neutral',
            'speak': True
        }
    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    # Run the Kivy mobile app
    SeeForMeApp().run()