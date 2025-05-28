import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .container import FrameData, DroneData
from .devices.tello import TelloDevice
from .. import logger


class NavigationManager:
    """Manages camera operations and frame processing.
    
    Responsibilities:
    1. Retrieve drone metadata and measurements
    2. has navigation movement history bank for return home
    2. Navigation algorithm
    """
    
    def __init__(self, device: TelloDevice):
        self._device = device
        self._frame_counter = 0
        self._last_telemetry: Optional[DroneData] = DroneData()
        self.waypoints_bank = None

    def get_measurements(self) -> Optional[DroneData]:
        """Get the next frame from the device with both raw and display versions."""
        try:
            # Get raw metadata from device
            self._last_telemetry.height = self._device.get_height()
            self._last_telemetry.battery = self._device.get_battery()
            self._last_telemetry.temperature = self._device.get_temperature()
            return self._last_telemetry
            
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    @property
    def last_telemetry(self) -> Optional[DroneData]:
        """Get the last captured frame data."""
        return self.last_telemetry
