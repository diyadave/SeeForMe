#!/usr/bin/env python3
"""
Advanced Camera Management System for Accessibility App
Handles seamless switching between front (webcam) and back (phone) cameras
Fixes: Unicode logging, orange screen, camera mapping, threading issues
"""

import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import platform
import sys
import os


import io
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Configure logging with emoji support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class CameraManager:
    """Advanced camera management with front/back switching for accessibility app"""
    
    def __init__(self):
        logger.info("📷 Initializing Camera Manager...")
        
        # Fixed camera configuration for your setup
        self.cameras = {}
        self.current_camera = None
        self.current_source = 'front'  # 'front' or 'back'
        self.is_streaming = False
        
        # Camera settings optimized for accessibility
        self.default_width = 640
        self.default_height = 480
        self.default_fps = 30
        
        # Threading for smooth operation
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=5)  # Increased buffer
        self.shutdown_event = threading.Event()
        self.camera_lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.dropped_frames = 0
        
        # Callbacks for UI integration
        self.frame_callbacks = []
        
        # Error handling and recovery
        self.max_connection_retries = 3
        self.camera_init_timeout = 5
        self.frame_capture_timeout = 1.0
        
        # Initialize cameras with your specific setup
        self.setup_camera_configuration()
        self.discover_cameras()
        
        logger.info("✅ Camera Manager initialized successfully")
    
    def setup_camera_configuration(self):
        """Setup camera configuration for your specific hardware"""
        logger.info("🔧 Setting up camera configuration...")
        
        # Your specific camera mapping
        self.camera_mapping = {
            'front': 0,  # Your working webcam
            'back': 1    # Android IP webcam app camera
        }
        
        # Camera-specific settings
        self.camera_settings = {
            'front': {
                'name': 'Front Webcam',
                'width': 640,
                'height': 480,
                'fps': 30,
                'mirror': True,  # Mirror front camera for natural view
                'auto_exposure': True
            },
            'back': {
                'name': 'Back Camera (Phone)',
                'width': 640,
                'height': 480,
                'fps': 25,  # Slightly lower for network camera
                'mirror': False,
                'auto_exposure': True,
                'connection_timeout': 10  # Extra timeout for network camera
            }
        }
        
        logger.info(f"📋 Camera mapping: {self.camera_mapping}")
    
    def discover_cameras(self):
        """Discover and test available cameras"""
        logger.info("🔍 Discovering available cameras...")
        
        self.available_cameras = {}
        
        # Test specific camera indices
        for camera_type, camera_index in self.camera_mapping.items():
            try:
                logger.info(f"🔄 Testing {camera_type} camera at index {camera_index}...")
                
                # Test camera with timeout
                test_successful = self.test_camera_connection(camera_index, camera_type)
                
                if test_successful:
                    settings = self.camera_settings[camera_type]
                    camera_info = {
                        'index': camera_index,
                        'type': camera_type,
                        'name': settings['name'],
                        'width': settings['width'],
                        'height': settings['height'],
                        'fps': settings['fps'],
                        'status': 'available'
                    }
                    
                    self.available_cameras[camera_index] = camera_info
                    logger.info(f"✅ {camera_type} camera available: {settings['name']}")
                else:
                    logger.warning(f"⚠️ {camera_type} camera at index {camera_index} not available")
                    
            except Exception as e:
                logger.error(f"❌ Error testing {camera_type} camera: {str(e)}")
        
        if not self.available_cameras:
            logger.error("❌ No cameras found! Please check camera connections.")
            raise Exception("No cameras available")
        
        logger.info(f"📊 Found {len(self.available_cameras)} working cameras")
    
    def test_camera_connection(self, camera_index: int, camera_type: str) -> bool:
        """Test camera connection with proper timeout and error handling"""
        cap = None
        try:
            # Create capture object with backend preference
            if camera_type == 'back':
                # For network cameras, try different backends
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DirectShow on Windows
                if not cap.isOpened():
                    cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                return False
            
            # Set basic properties to avoid orange screen
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
            
            # Test frame capture with timeout
            start_time = time.time()
            frame_captured = False
            
            for attempt in range(5):  # Try multiple times
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Check if frame is not just orange/solid color
                    if self.is_valid_frame(frame):
                        frame_captured = True
                        break
                
                if time.time() - start_time > self.camera_init_timeout:
                    break
                    
                time.sleep(0.1)  # Short delay between attempts
            
            return frame_captured
            
        except Exception as e:
            logger.debug(f"Camera test error: {str(e)}")
            return False
        finally:
            if cap:
                cap.release()
    
    def is_valid_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is valid (not solid color/orange screen)"""
        if frame is None or frame.size == 0:
            return False
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate standard deviation - low std indicates solid color
        std_dev = np.std(gray)
        
        # Check for minimum variance (threshold for "interesting" content)
        if std_dev < 10:  # Very low variance = likely solid color
            return False
        
        # Check for reasonable brightness range
        mean_brightness = np.mean(gray)
        if mean_brightness < 5 or mean_brightness > 250:  # Too dark or too bright
            return False
        
        return True
    
    def switch_camera(self, camera_type: str = 'front') -> bool:
        """Switch to specified camera type with proper error handling"""
        if camera_type not in ['front', 'back']:
            logger.error(f"❌ Invalid camera type: {camera_type}")
            return False
        
        with self.camera_lock:
            if camera_type == self.current_source and self.current_camera is not None:
                logger.info(f"📷 Already using {camera_type} camera")
                return True
            
            logger.info(f"🔄 Switching to {camera_type} camera...")
            
            try:
                # Stop current streaming
                was_streaming = self.is_streaming
                if was_streaming:
                    self.stop_streaming()
                    time.sleep(0.5)  # Allow time for cleanup
                
                # Release current camera safely
                self.release_current_camera()
                
                # Initialize new camera
                success = self.initialize_camera(camera_type)
                
                if success:
                    logger.info(f"✅ Successfully switched to {camera_type} camera")
                    
                    # Resume streaming if it was active
                    if was_streaming:
                        self.start_streaming()
                    
                    return True
                else:
                    logger.error(f"❌ Failed to switch to {camera_type} camera")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Camera switch error: {str(e)}")
                return False
    
    def release_current_camera(self):
        """Safely release current camera"""
        if self.current_camera:
            try:
                self.current_camera.release()
                time.sleep(0.2)  # Give time for proper release
            except Exception as e:
                logger.debug(f"Camera release warning: {str(e)}")
            finally:
                self.current_camera = None
    
    def initialize_camera(self, camera_type: str) -> bool:
        """Initialize camera with proper configuration"""
        camera_index = self.camera_mapping.get(camera_type)
        if camera_index is None:
            logger.error(f"❌ No mapping for {camera_type} camera")
            return False
        
        if camera_index not in self.available_cameras:
            logger.error(f"❌ {camera_type} camera not available")
            return False
        
        try:
            settings = self.camera_settings[camera_type]
            
            # Create camera capture
            if camera_type == 'back':
                # Special handling for network cameras
                new_camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if not new_camera.isOpened():
                    new_camera = cv2.VideoCapture(camera_index)
            else:
                new_camera = cv2.VideoCapture(camera_index)
            
            if not new_camera.isOpened():
                logger.error(f"❌ Failed to open {camera_type} camera")
                return False
            
            # Configure camera settings
            self.configure_camera(new_camera, settings)
            
            # Test the camera with multiple attempts
            frame_test_success = False
            for attempt in range(3):
                ret, frame = new_camera.read()
                if ret and frame is not None and self.is_valid_frame(frame):
                    frame_test_success = True
                    break
                time.sleep(0.2)
            
            if not frame_test_success:
                logger.error(f"❌ {camera_type} camera not producing valid frames")
                new_camera.release()
                return False
            
            # Update current camera
            self.current_camera = new_camera
            self.current_source = camera_type
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera initialization error: {str(e)}")
            return False
    
    def configure_camera(self, camera: cv2.VideoCapture, settings: Dict):
        """Configure camera with optimal settings"""
        try:
            # Basic resolution and FPS
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
            camera.set(cv2.CAP_PROP_FPS, settings['fps'])
            
            # Performance optimizations
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Auto exposure and focus if supported
            if settings.get('auto_exposure', False):
                try:
                    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust exposure
                except:
                    pass
            
            # Additional settings for stability
            try:
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                camera.set(cv2.CAP_PROP_BRIGHTNESS, 0)
                camera.set(cv2.CAP_PROP_CONTRAST, 32)
                camera.set(cv2.CAP_PROP_SATURATION, 32)
            except:
                pass  # Some properties may not be supported
            
            logger.debug(f"📷 Camera configured: {settings['width']}x{settings['height']} @ {settings['fps']}fps")
            
        except Exception as e:
            logger.warning(f"⚠️ Camera configuration warning: {str(e)}")
    
    def start_streaming(self) -> bool:
        """Start continuous camera streaming with error handling"""
        if self.is_streaming:
            logger.info("📹 Streaming already active")
            return True
        
        if not self.current_camera:
            logger.error("❌ No camera available for streaming")
            return False
        
        logger.info(f"📹 Starting {self.current_source} camera streaming...")
        
        try:
            self.is_streaming = True
            self.shutdown_event.clear()
            
            # Clear any old frames
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_worker,
                daemon=True,
                name=f"CameraCapture-{self.current_source}"
            )
            self.capture_thread.start()
            
            logger.info("✅ Camera streaming started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start streaming: {str(e)}")
            self.is_streaming = False
            return False
    
    def stop_streaming(self):
        """Stop camera streaming safely"""
        if not self.is_streaming:
            return
        
        logger.info("🛑 Stopping camera streaming...")
        
        self.is_streaming = False
        self.shutdown_event.set()
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
            if self.capture_thread.is_alive():
                logger.warning("⚠️ Capture thread did not stop gracefully")
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ Camera streaming stopped")
    
    def _capture_worker(self):
        """Worker thread for continuous frame capture with error recovery"""
        logger.debug(f"🎬 Capture worker started for {self.current_source}")
        
        fps_start_time = time.time()
        local_frame_count = 0
        consecutive_failures = 0
        max_failures = 10
        
        while not self.shutdown_event.is_set() and self.is_streaming:
            try:
                with self.camera_lock:
                    if not self.current_camera or not self.current_camera.isOpened():
                        logger.error("❌ Camera not available in capture worker")
                        break
                    
                    # Capture frame with timeout
                    ret, frame = self.current_camera.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.debug(f"⚠️ Frame capture failed (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("❌ Too many consecutive frame capture failures")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Validate frame
                if not self.is_valid_frame(frame):
                    consecutive_failures += 1
                    logger.debug(f"⚠️ Invalid frame received (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("❌ Too many invalid frames")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add to queue (non-blocking)
                frame_data = {
                    'frame': processed_frame,
                    'timestamp': time.time(),
                    'camera_type': self.current_source,
                    'frame_id': local_frame_count
                }
                
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                    
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        self.dropped_frames += 1
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(processed_frame, self.current_source)
                    except Exception as e:
                        logger.error(f"❌ Frame callback error: {str(e)}")
                
                # Update FPS counter
                local_frame_count += 1
                if local_frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - fps_start_time
                    if elapsed > 0:
                        fps = 30 / elapsed
                        self.fps_counter = fps
                        fps_start_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"❌ Capture worker error: {str(e)}")
                
                if consecutive_failures >= max_failures:
                    logger.error("❌ Too many capture worker errors")
                    break
                
                time.sleep(0.1)
        
        logger.debug(f"🎬 Capture worker stopped for {self.current_source}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process captured frame with camera-specific adjustments"""
        if frame is None:
            return frame
        
        try:
            # Mirror front camera frames for natural selfie view
            settings = self.camera_settings.get(self.current_source, {})
            if settings.get('mirror', False):
                frame = cv2.flip(frame, 1)
            
            # Additional processing for network cameras
            if self.current_source == 'back':
                # Slight noise reduction for network cameras
                frame = cv2.bilateralFilter(frame, 5, 80, 80)
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Frame processing error: {str(e)}")
            return frame
    
    def get_latest_frame(self) -> Optional[Dict]:
        """Get the latest captured frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from current camera"""
        if not self.current_camera or not self.current_camera.isOpened():
            logger.error("❌ No camera available for capture")
            return None
        
        try:
            with self.camera_lock:
                ret, frame = self.current_camera.read()
                
            if ret and frame is not None and self.is_valid_frame(frame):
                return self.process_frame(frame)
            else:
                logger.warning("⚠️ Failed to capture valid frame")
                return None
                
        except Exception as e:
            logger.error(f"❌ Frame capture error: {str(e)}")
            return None
    
    def switch_to_emotion_camera(self) -> bool:
        """Switch to camera for emotion detection (front camera)"""
        return self.switch_camera('front')
    
    def switch_to_scene_camera(self) -> bool:
        """Switch to camera for scene analysis (back camera)"""
        return self.switch_camera('back')
    
    def get_camera_info(self) -> Dict:
        """Get comprehensive camera information"""
        try:
            if not self.current_camera:
                return {'status': 'no_camera', 'available_cameras': len(self.available_cameras)}
            
            camera_index = self.camera_mapping.get(self.current_source)
            camera_info = self.available_cameras.get(camera_index, {})
            
            return {
                'status': 'streaming' if self.is_streaming else 'ready',
                'current_source': self.current_source,
                'camera_index': camera_index,
                'camera_name': camera_info.get('name', 'Unknown'),
                'width': int(self.current_camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.current_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': round(self.fps_counter, 1),
                'frame_count': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'available_cameras': len(self.available_cameras),
                'queue_size': self.frame_queue.qsize()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting camera info: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def is_camera_ready(self) -> bool:
        """Check if camera is ready for use"""
        return (self.current_camera is not None and 
                self.current_camera.isOpened() and 
                not self.shutdown_event.is_set())
    
    def cleanup(self):
        """Clean up camera resources"""
        logger.info("🧹 Cleaning up camera manager...")
        
        # Stop streaming
        self.stop_streaming()
        
        # Release current camera
        with self.camera_lock:
            self.release_current_camera()
        
        # Clear callbacks
        self.frame_callbacks.clear()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ Camera manager cleanup complete")
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass


# Testing and utility functions
def test_camera_system():
    """Test the camera system thoroughly"""
    try:
        print("🧪 Testing Camera System...")
        print("=" * 50)
        
        manager = CameraManager()
        
        # Display available cameras
        print(f"📋 Available cameras: {len(manager.available_cameras)}")
        for idx, info in manager.available_cameras.items():
            print(f"   {idx}: {info['name']} ({info['type']})")
        
        # Test switching
        print("\n🔄 Testing camera switching...")
        
        # Test front camera
        if manager.switch_camera('front'):
            print("✅ Front camera switch successful")
            manager.start_streaming()
            time.sleep(2)
            
            # Test back camera
            if manager.switch_camera('back'):
                print("✅ Back camera switch successful")
                time.sleep(2)
            
            manager.stop_streaming()
        
        # Display final info
        info = manager.get_camera_info()
        print(f"\n📊 Final Status: {info}")
        
        manager.cleanup()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        logger.error(f"Test error: {str(e)}")


if __name__ == "__main__":
    test_camera_system()