import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .devices.tello import TelloDevice


@dataclass
class FrameData:
    """Container for frame data and metadata."""
    raw_frame: Optional[np.ndarray] = None          # Original frame for processing
    display_frame: Optional[np.ndarray] = None      # Frame with overlays for display
    timestamp: float = 0.0
    resolution: Tuple[int, int] = (960, 720)
    frame_number: int = 0

class CameraManager:
    """Manages camera operations and frame processing.
    
    Responsibilities:
    1. Frame acquisition from drone
    2. Basic frame preprocessing
    3. Adding visual overlays for display
    4. Managing frame buffers
    """
    
    def __init__(self, device: TelloDevice):
        self._device = device
        self._frame_counter = 0
        self._last_frame: Optional[FrameData] = None
        
    def get_frame(self) -> Optional[FrameData]:
        """Get the next frame from the device with both raw and display versions."""
        try:
            # Get raw frame from device
            raw_frame = self._device.get_frame()
            if raw_frame is None:
                return None
                
            # Create display frame (copy for overlays)
            display_frame = raw_frame.copy()
            
            # Update frame data
            self._frame_counter += 1
            frame_data = FrameData(
                raw_frame=raw_frame,
                display_frame=display_frame,
                frame_number=self._frame_counter
            )
            self._last_frame = frame_data
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None
            
    def add_overlay(self, frame_data: FrameData, overlay_info: Dict[str, Any]) -> None:
        """Add visual overlays to the display frame."""
        if frame_data.display_frame is None:
            return
            
        try:
            frame = frame_data.display_frame
            
            # Draw detected faces
            if "detected_faces" in overlay_info:
                for face in overlay_info["detected_faces"]:
                    bbox = face.get("bbox")
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Add confidence score if available
                        confidence = face.get("confidence", 0)
                        cv2.putText(frame, f"{confidence:.2f}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 1)
            
            # Add mission info overlay
            if "mission_info" in overlay_info:
                info = overlay_info["mission_info"]
                y_pos = 30
                for key, value in info.items():
                    text = f"{key}: {value}"
                    cv2.putText(frame, text, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 20
                    
            # Add state info
            if "state" in overlay_info:
                state_text = f"State: {overlay_info['state']}"
                cv2.putText(frame, state_text, (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        except Exception as e:
            logger.error(f"Error adding overlay: {str(e)}")
            
    @property
    def last_frame(self) -> Optional[FrameData]:
        """Get the last captured frame data."""
        return self._last_frame
