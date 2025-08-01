from app import app, socketio
import logging

logging.basicConfig(level=logging.INFO)

# Initialize database tables
with app.app_context():
    from app import db
    import models  # Import models to register them
    db.create_all()
    print("💾 Database initialized with memory tables")

if __name__ == '__main__':
    print("🚀 Starting SeeForMe Assistant with Memory System...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, log_output=True)
