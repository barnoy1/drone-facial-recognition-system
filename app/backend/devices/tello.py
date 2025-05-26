from abc import ABC, abstractmethod
import time
import cv2
import numpy as np
from typing import Optional

from app.backend import ConfigManager


class TelloDevice(ABC):
    """Abstract base class for Tello device implementations."""

    def __init__(self):
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        pass
    
    @abstractmethod
    def takeoff(self) -> bool:
        pass
    
    @abstractmethod
    def land(self) -> bool:
        pass
    
    @abstractmethod
    def emergency(self) -> bool:
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        pass

class WebcamMockTello(TelloDevice):
    """Mock Tello implementation using webcam as video source."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def __init__(self):
        self.cap = None
        self.is_connected = False
        self.is_flying = False
        self._device_index = 0
        self._frame_size = (960, 720)  # Match Tello camera resolution
    
    def connect(self) -> bool:
        if self.is_connected:
            print('Already connected to webcam')
            return True

        print("Attempting to connect to webcam...")
        for attempt in range(self.MAX_RETRIES):
            try:
                # Try multiple devices
                for device_id in range(2):
                    try:
                        cap = cv2.VideoCapture(device_id)
                        if cap is not None and cap.isOpened():
                            # Test frame capture
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                # Configure camera properties
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_size[0])
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_size[1])
                                cap.set(cv2.CAP_PROP_FPS, 30)
                                
                                if self.cap:
                                    self.cap.release()
                                self.cap = cap
                                self._device_index = device_id
                                self.is_connected = True
                                print(f"Successfully connected to camera {device_id}")
                                return True
                            else:
                                cap.release()
                    except Exception as e:
                        print(f"Failed to initialize camera {device_id}: {e}")
                        if cap:
                            cap.release()
                
                # If no camera found in this attempt, wait before retry
                if attempt < self.MAX_RETRIES - 1:
                    print(f"No working camera found, retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                    
            except Exception as e:
                print(f"Error during camera connection attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
        
        print("Failed to connect to any camera after maximum retries")
        self.is_connected = False
        return False
    
    def disconnect(self) -> bool:
        """Safely disconnect from webcam."""
        try:
            if self.cap:
                self.cap.release()
            self.cap = None 
            self.is_connected = False
            self.is_flying = False
            return True
        except Exception as e:
            print(f"Error disconnecting from camera: {e}")
            return False
    
    def takeoff(self) -> bool:
        print("Mock takeoff")
        self.is_flying = True
        return True
    
    def land(self) -> bool:
        print("Mock landing")
        self.is_flying = False
        return True
    
    def emergency(self) -> bool:
        print("Mock emergency stop")
        self.is_flying = False
        if self.cap:
            self.cap.release()
            self.cap = None
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from webcam with error handling."""
        if not self.cap or not self.is_connected:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to read frame")
                return None
                
            # Match Tello camera resolution
            if frame.shape[:2] != self._frame_size[::-1]:
                frame = cv2.resize(frame, self._frame_size)
                
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.disconnect()

class FolderMockTello(TelloDevice):
    """Mock Tello implementation using image folder as video source."""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.image_files = []
        self.current_idx = 0
        self.is_connected = False
        self.is_flying = False
        
    def connect(self) -> bool:
        import os
        import glob
        
        try:
            self.image_files = sorted(glob.glob(os.path.join(self.folder_path, "*.jpg")))
            if not self.image_files:
                print("No images found in folder")
                return False
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to image folder: {e}")
            return False
    
    def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    def takeoff(self) -> bool:
        self.is_flying = True
        return True
    
    def land(self) -> bool:
        self.is_flying = False
        return True
    
    def emergency(self) -> bool:
        self.is_flying = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.image_files:
            return None
            
        image_path = self.image_files[self.current_idx]
        frame = cv2.imread(image_path)
        if frame is None:
            return None
            
        # Cycle through images
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class RealTello(TelloDevice):
    """Real Tello drone implementation."""
    
    def __init__(self):
        try:
            from djitellopy import Tello
            self.drone = Tello()
            self.is_flying = False
        except ImportError:
            raise ImportError("djitellopy package is required for real Tello support")
    
    def connect(self) -> bool:
        try:
            self.drone.connect()
            self.drone.streamon()
            return True
        except Exception as e:
            print(f"Failed to connect to Tello: {e}")
            return False
    
    def disconnect(self) -> bool:
        try:
            self.drone.streamoff()
            return True
        except Exception as e:
            print(f"Failed to disconnect from Tello: {e}")
            return False
    
    def takeoff(self) -> bool:
        try:
            self.drone.takeoff()
            self.is_flying = True
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        try:
            self.drone.land()
            self.is_flying = False
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False
    
    def emergency(self) -> bool:
        try:
            self.drone.emergency()
            self.is_flying = False
            return True
        except Exception as e:
            print(f"Emergency stop failed: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        try:
            frame = self.drone.get_frame_read().frame
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to get frame: {e}")
            return None

class TelloFactory:
    """Factory class to create appropriate Tello device implementation."""
    
    @staticmethod
    def create_tello() -> TelloDevice:
        config = ConfigManager

        try:
            # Wait for config to be loaded
            if not config.config:
                raise RuntimeError("Configuration not loaded. Call load_config() first.")
            
            mock_enabled = config.tello_config.mock_enabled
            mock_source = config.tello_config.mock_source
            mock_folder_path = config.tello_config.mock_folder_path
            
            print(f"Config values: mock_enabled={mock_enabled}, mock_source={mock_source}")
            
            if mock_enabled:
                if mock_source == "webcam":
                    print("Creating WebcamMockTello")
                    device = WebcamMockTello()
                elif mock_source == "folder":
                    print("Creating FolderMockTello")
                    device = FolderMockTello(mock_folder_path)
                else:
                    raise ValueError(f"Unknown mock source: {mock_source}")
            else:
                print("Creating RealTello")
                device = RealTello()
            
            # Test connection
            print(f"Connecting to {device.__class__.__name__}")
            if not device.connect():
                raise RuntimeError(f"Failed to connect to {device.__class__.__name__}")
            return device
            
        except Exception as e:
            print(f"Error creating Tello device: {str(e)}")
            raise
