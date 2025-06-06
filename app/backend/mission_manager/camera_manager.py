import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from app.backend.container import FrameData, HUD, MissionState
from app.backend.devices.tello import TelloDevice
logger = logging.getLogger("app_logger")
from PIL import Image, ImageDraw, ImageFont


def present_frame(pil_image: Image.Image, save_path: str = None) -> None:
    """Present the PIL Image by saving to file or displaying."""
    try:
        if save_path:
            pil_image.save(save_path)
            logger.error(f"Image saved to {save_path}")
        else:
            # Display the image (works in local environments with GUI support)
            pil_image.show()
    except Exception as e:
        logger.error(f"Error presenting image: {e}")


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
            logger.error("Error: mission_state.frame_data is None")
            return

        # Convert frame_data to NumPy array and validate
        try:
            frame_array = np.asarray(mission_state.frame_data.display_frame, dtype=np.uint8)
            if frame_array.shape != (720, 960, 3) or frame_array.dtype != np.uint8:
                raise ValueError(f"Invalid frame: shape={frame_array.shape}, dtype={frame_array.dtype}")
        except Exception as e:
            logger.error(f"Error converting frame_data to NumPy array: {e}")
            return

        # Convert BGR (OpenCV) to RGB (PIL)
        try:
            pil_image = Image.fromarray(frame_array)  # Fixed: Use rgb_array instead of frame_array
        except Exception as e:
            logger.error(f"Error converting to PIL Image: {e}")
            return

        # Create draw object for text overlays
        draw = ImageDraw.Draw(pil_image, 'RGBA')  # Use RGBA mode for transparency

        # Prepare text overlays
        texts = []
        try:
            texts.append(f"State: {mission_state.pipeline_current_node.name}")
        except AttributeError as e:
            logger.error(f"Error accessing pipeline_state.name: {e}")
            texts.append("State: Unknown")

        if mission_state.frame_data:
            try:
                texts.append(f"Frame ID: {mission_state.frame_data.frame_number}")
            except AttributeError as e:
                logger.error(f"Error accessing drone_data: {e}")

        texts.append(f"FPS: {mission_state.fps}Hz")

        # Calculate rectangle bounds to encompass all text
        padding = self.hud.PADDING
        y_offset = self.hud.Y_OFFSET
        line_spacing = self.hud.LINE_SPACING
        text_heights = [self.hud.font.getbbox(text)[3] for text in texts]  # Height of each text line
        total_text_height = sum(text_heights) + (len(texts) - 1) * line_spacing
        max_text_width = max(self.hud.font.getbbox(text)[2] for text in texts)  # Widest text

        # Define rectangle coordinates
        rect_x0 = padding - 5
        rect_y0 = y_offset - 5
        rect_x1 = padding + max_text_width + 5
        rect_y1 = y_offset + total_text_height + 5

        # Draw semi-transparent rectangle (RGBA: 0-255, alpha=100 for moderate transparency)
        draw.rectangle(
            (rect_x0, rect_y0, rect_x1, rect_y1),
            fill=(0, 0, 0, 100)  # Black with transparency
        )

        # Render text overlays using HUD parameters
        for i, text in enumerate(texts):
            try:
                draw.text(
                    (padding, y_offset + i * line_spacing),
                    text,
                    font=self.hud.font,
                    fill=self.hud.FONT_COLOR
                )
            except Exception as e:
                logger.error(f"Error rendering text overlay '{text}': {e}")

        # Store the PIL Image with overlays
        mission_state.frame_data.display_frame = pil_image

    @property
    def last_frame(self) -> Optional[FrameData]:
        """Get the last captured frame data."""
        return self._last_frame
