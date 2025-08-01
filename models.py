"""
Database models for SeeForMe memory system
Stores conversation history, emotions, and face recognition data
"""
from datetime import datetime
from app import db

class UserSession(db.Model):
    """Track user sessions and emotional states over time"""
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=False, default="friend")
    session_date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    emotional_state = db.Column(db.String(50), nullable=True)  # sad, happy, angry, etc.
    conversation_summary = db.Column(db.Text, nullable=True)
    last_interaction = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserSession {self.user_name} - {self.session_date}>'

class ConversationHistory(db.Model):
    """Store detailed conversation history"""
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=False, default="friend")
    user_input = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    emotion_detected = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Conversation {self.user_name} - {self.timestamp}>'

class FaceRecognition(db.Model):
    """Store face encodings and associated names for person recognition"""
    id = db.Column(db.Integer, primary_key=True)
    person_name = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text, nullable=False)  # JSON string of face encoding
    first_seen = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    interaction_count = db.Column(db.Integer, default=1)
    relationship_notes = db.Column(db.Text, nullable=True)  # friend, family, etc.
    
    def __repr__(self):
        return f'<FaceRecognition {self.person_name}>'

class EmotionalMemory(db.Model):
    """Track emotional patterns and provide continuity across sessions"""
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=False, default="friend")
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    dominant_emotion = db.Column(db.String(50), nullable=False)
    emotion_intensity = db.Column(db.Float, default=0.5)  # 0.0 to 1.0
    context_description = db.Column(db.Text, nullable=True)
    resolution_status = db.Column(db.String(50), default="unresolved")  # resolved, ongoing, unresolved
    
    def __repr__(self):
        return f'<EmotionalMemory {self.user_name} - {self.date} - {self.dominant_emotion}>'