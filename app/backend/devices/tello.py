from abc import ABC, abstractmethod
import time
import cv2
import numpy as np
from typing import Optional

from app import logger
from app.backend import ConfigManager
from app.core.utilities.decorators.singleton import Singleton


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

    @abstractmethod
    def get_height(self):
        pass

    @abstractmethod
    def get_battery(self):
        pass

    @abstractmethod
    def get_temperature(self):
        pass



@Singleton
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
        self.cap = None

    def connect(self) -> bool:
        if self.is_connected:
            logger.info('Already connected to webcam')
            return True

        logger.info("Attempting to connect to webcam...")
        for attempt in range(self.MAX_RETRIES):
            try:
                # Try multiple devices
                for device_id in range(2):
                    try:
                        self.cap = cv2.VideoCapture(device_id)
                        if self.cap is not None and self.cap.isOpened():
                            # Test frame capture
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                # Configure camera properties
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_size[0])
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_size[1])
                                self.cap.set(cv2.CAP_PROP_FPS, 30)

                                # if self.cap:
                                #     self.cap.release()
                                self._device_index = device_id
                                self.is_connected = True
                                logger.info(f"Successfully connected to camera {device_id}")
                                return True
                            else:
                                self.cap.release()
                    except Exception as e:
                        logger.info(f"Failed to initialize camera {device_id}: {e}")
                        if self.cap:
                            self.cap.release()

                # If no camera found in this attempt, wait before retry
                if attempt < self.MAX_RETRIES - 1:
                    logger.info(f"No working camera found, retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)

            except Exception as e:
                logger.info(f"Error during camera connection attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

        logger.info("Failed to connect to any camera after maximum retries")
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
            logger.info(f"Error disconnecting from camera: {e}")
            return False

    def takeoff(self) -> bool:
        logger.info("Mock takeoff")
        self.is_flying = True
        return True

    def land(self) -> bool:
        logger.info("Mock landing")
        self.is_flying = False
        return True

    def emergency(self) -> bool:
        logger.info("Mock emergency stop")
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
                logger.info("Failed to read frame")
                return None

            # Match Tello camera resolution
            if frame.shape[:2] != self._frame_size[::-1]:
                frame = cv2.resize(frame, self._frame_size)

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.info(f"Error capturing frame: {e}")
            return None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.disconnect()


    def get_height(self):
        return 0

    def get_battery(self):
        return 0.3

    def get_temperature(self):
        return 0


@Singleton
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
                logger.info("No images found in folder")
                return False
            self.is_connected = True
            return True
        except Exception as e:
            logger.info(f"Failed to connect to image folder: {e}")
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

    def get_height(self):
        return 0

    def get_battery(self):
        return 0.3

    def get_temperature(self):
        return 0


@Singleton
class DJITello(TelloDevice):
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
            logger.info(f"Failed to connect to Tello: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            self.drone.streamoff()
            return True
        except Exception as e:
            logger.info(f"Failed to disconnect from Tello: {e}")
            return False

    def takeoff(self) -> bool:
        try:
            self.drone.takeoff()
            self.is_flying = True
            return True
        except Exception as e:
            logger.info(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        try:
            self.drone.land()
            self.is_flying = False
            return True
        except Exception as e:
            logger.info(f"Landing failed: {e}")
            return False

    def emergency(self) -> bool:
        try:
            self.drone.emergency()
            self.is_flying = False
            return True
        except Exception as e:
            logger.info(f"Emergency stop failed: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            frame = self.drone.get_frame_read().frame
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.info(f"Failed to get frame: {e}")
            return None

    def get_height(self):
        return 0

    def get_battery(self):
        return 0.3

    def get_temperature(self):
        return 0


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

            logger.info(f"Config values: mock_enabled={mock_enabled}, mock_source={mock_source}")

            if mock_enabled:
                if mock_source == "webcam":
                    logger.info("Creating WebcamMockTello")
                    device = WebcamMockTello.instance()
                elif mock_source == "folder":
                    logger.info("Creating FolderMockTello")
                    device = FolderMockTello(mock_folder_path).instance()
                else:
                    raise ValueError(f"Unknown mock source: {mock_source}")
            else:
                logger.info("Creating RealTello")
                device = DJITello().instance()

            # Test connection
            logger.info(f"Connecting to {device.__class__.__name__}")
            if not device.connect():
                raise RuntimeError(f"Failed to connect to {device.__class__.__name__}")
            return device

        except Exception as e:
            logger.info(f"Error creating Tello device: {str(e)}")
            raise
