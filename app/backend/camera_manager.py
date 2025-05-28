import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .container import FrameData, HUD, MissionState
from .devices.tello import TelloDevice
from .. import logger
from PIL import Image, ImageDraw, ImageFont


def present_frame(pil_image: Image.Image, save_path: str = None) -> None:
    """Present the PIL Image by saving to file or displaying."""
    try:
        if save_path:
            pil_image.save(save_path)
            print(f"Image saved to {save_path}")
        else:
            # Display the image (works in local environments with GUI support)
            pil_image.show()
    except Exception as e:
        print(f"Error presenting image: {e}")


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
        self.hud = HUD()

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

    def add_overlay(self, mission_state: MissionState) -> None:
        """Add overlays to the frame based on current state and store as PIL Image."""
        if mission_state.frame_data is None:
            print("Error: mission_state.frame_data is None")
            return

        # Convert frame_data to NumPy array and validate
        try:
            frame_array = np.asarray(mission_state.frame_data.display_frame, dtype=np.uint8)
            if frame_array.shape != (720, 960, 3) or frame_array.dtype != np.uint8:
                raise ValueError(f"Invalid frame: shape={frame_array.shape}, dtype={frame_array.dtype}")
        except Exception as e:
            print(f"Error converting frame_data to NumPy array: {e}")
            return

        # Convert BGR (OpenCV) to RGB (PIL)
        try:
            rgb_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_array)
        except Exception as e:
            print(f"Error converting to PIL Image: {e}")
            return

        # Create draw object for text overlays
        draw = ImageDraw.Draw(pil_image)

        # Prepare text overlays
        texts = []
        try:
            texts.append(f"State: {mission_state.pipeline_state.name}")
        except AttributeError as e:
            print(f"Error accessing pipeline_state.name: {e}")
            texts.append("State: Unknown")

        texts.append(f"FPS: {mission_state.fps}Hz")

        if mission_state.detected_faces:
            try:
                texts.append(f"Faces: {len(mission_state.detected_faces)}")
            except Exception as e:
                print(f"Error processing detected_faces: {e}")

        if mission_state.drone_data:
            try:
                texts.append(f"Battery: {mission_state.drone_data.battery}%")
                texts.append(f"Height: {mission_state.drone_data.height:.1f}m")
            except AttributeError as e:
                print(f"Error accessing drone_data: {e}")


        # Render text overlays using HUD parameters
        for i, text in enumerate(texts):
            try:
                draw.text(
                    (self.hud.PADDING, self.hud.Y_OFFSET + i * self.hud.LINE_SPACING),
                    text,
                    font=self.hud.font,
                    fill=self.hud.FONT_COLOR
                )
            except Exception as e:
                print(f"Error rendering text overlay '{text}': {e}")

        # Store the PIL Image with overlays
        mission_state.frame_data.display_frame = pil_image

    @property
    def last_frame(self) -> Optional[FrameData]:
        """Get the last captured frame data."""
        return self._last_frame
