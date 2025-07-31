#!/usr/bin/env python3
"""
Camera Switcher - Manages front/back camera switching
Handles camera access and frame capture for analysis
"""

import logging
import threading
import time
import queue
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class CameraSwitcher:
    """Manages camera switching and frame capture"""
    
    def __init__(self):
        self.is_initialized = False
        self.permission_granted = False
        self.current_mode = 'front'  # 'front' or 'back'
        
        # Camera objects
        self.front_camera = None
        self.back_camera = None
        self.active_camera = None
        
        # Frame capture
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.stop_event = threading.Event()
        
        # Camera indices (may need adjustment based on system)
        self.front_camera_index = 0  # Usually webcam/front camera
        self.back_camera_index = 1   # Usually external/back camera if available
        
        logger.info("📹 Camera switcher initialized")
    
    def set_permission(self, permitted: bool):
        """Set camera permission status"""
        self.permission_granted = permitted
        if permitted:
            self.initialize_cameras()
        else:
            self.cleanup_cameras()
    
    def initialize_cameras(self):
        """Initialize available cameras"""
        if not self.permission_granted:
            logger.warning("⚠️ Camera permission not granted")
            return False
        
        try:
            # Try to initialize front camera (webcam)
            self.front_camera = cv2.VideoCapture(self.front_camera_index)
            if not self.front_camera.isOpened():
                logger.warning("⚠️ Front camera not available")
                self.front_camera = None
            else:
                # Configure front camera
                self.front_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.front_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.front_camera.set(cv2.CAP_PROP_FPS, 30)
                logger.info("✅ Front camera initialized")
            
            # Try to initialize back camera (if available)
            self.back_camera = cv2.VideoCapture(self.back_camera_index)
            if not self.back_camera.isOpened():
                logger.info("ℹ️ Back camera not available (using front camera)")
                self.back_camera = None
            else:
                # Configure back camera
                self.back_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.back_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.back_camera.set(cv2.CAP_PROP_FPS, 30)
                logger.info("✅ Back camera initialized")
            
            # Set active camera to front by default
            self.active_camera = self.front_camera
            self.is_initialized = True
            
            # Start capture thread
            self.start_capture_thread()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            return False
    
    def start_capture_thread(self):
        """Start continuous frame capture"""
        if self.capture_thread and self.capture_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("🎬 Camera capture thread started")
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        while not self.stop_event.is_set():
            if self.active_camera and self.active_camera.isOpened():
                try:
                    ret, frame = self.active_camera.read()
                    if ret:
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                except Exception as e:
                    logger.error(f"❌ Frame capture error: {e}")
            
            time.sleep(1/30)  # 30 FPS
    
    def switch_to(self, mode: str) -> bool:
        """Switch to specified camera mode"""
        if not self.is_initialized or not self.permission_granted:
            logger.warning("⚠️ Camera not initialized or permission denied")
            return False
        
        if mode not in ['front', 'back']:
            logger.error(f"❌ Invalid camera mode: {mode}")
            return False
        
        # If switching to back camera but it's not available, use front
        if mode == 'back' and self.back_camera is None:
            logger.info("ℹ️ Back camera not available, using front camera")
            mode = 'front'
        
        # Switch active camera
        if mode == 'front' and self.front_camera:
            self.active_camera = self.front_camera
            self.current_mode = 'front'
            logger.info("📱 Switched to front camera")
            return True
        elif mode == 'back' and self.back_camera:
            self.active_camera = self.back_camera
            self.current_mode = 'back'
            logger.info("📷 Switched to back camera")
            return True
        else:
            logger.error(f"❌ Cannot switch to {mode} camera - not available")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture current frame"""
        if not self.is_initialized or not self.permission_granted:
            return None
        
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        
        return None
    
    def capture_high_res_frame(self) -> Optional[np.ndarray]:
        """Capture a high-resolution frame for analysis"""
        if not self.active_camera or not self.active_camera.isOpened():
            return None
        
        try:
            # Temporarily increase resolution for analysis
            self.active_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.active_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ret, frame = self.active_camera.read()
            
            # Reset to normal resolution
            self.active_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.active_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if ret:
                return frame
                
        except Exception as e:
            logger.error(f"❌ High-res capture failed: {e}")
        
        return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get information about available cameras"""
        info = {
            'front_available': self.front_camera is not None,
            'back_available': self.back_camera is not None,
            'current_mode': self.current_mode,
            'resolution': (640, 480),
            'fps': 30
        }
        
        if self.active_camera and self.active_camera.isOpened():
            try:
                width = int(self.active_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.active_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.active_camera.get(cv2.CAP_PROP_FPS))
                info.update({
                    'resolution': (width, height),
                    'fps': fps
                })
            except Exception as e:
                logger.warning(f"⚠️ Could not get camera info: {e}")
        
        return info
    
    def test_camera_switching(self) -> Dict[str, bool]:
        """Test camera switching functionality"""
        results = {
            'front_camera_test': False,
            'back_camera_test': False,
            'switching_test': False
        }
        
        # Test front camera
        if self.switch_to('front'):
            frame = self.capture_frame()
            results['front_camera_test'] = frame is not None
        
        # Test back camera
        if self.switch_to('back'):
            frame = self.capture_frame()
            results['back_camera_test'] = frame is not None
        
        # Test switching
        if self.switch_to('front') and self.switch_to('back'):
            results['switching_test'] = True
        
        logger.info(f"📋 Camera test results: {results}")
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current camera status"""
        return {
            'status': 'active' if self.is_initialized else 'inactive',
            'permission': self.permission_granted,
            'current_mode': self.current_mode,
            'cameras_available': {
                'front': self.front_camera is not None,
                'back': self.back_camera is not None
            },
            'capture_active': self.capture_thread and self.capture_thread.is_alive(),
            'info': self.get_camera_info()
        }
    
    def cleanup_cameras(self):
        """Clean up camera resources"""
        logger.info("🧹 Cleaning up cameras...")
        
        # Stop capture thread
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Release cameras
        if self.front_camera:
            self.front_camera.release()
            self.front_camera = None
        
        if self.back_camera:
            self.back_camera.release()
            self.back_camera = None
        
        self.active_camera = None
        self.is_initialized = False
        
        logger.info("✅ Camera cleanup completed")
    
    def cleanup(self):
        """Full cleanup"""
        self.cleanup_cameras()