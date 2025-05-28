from dataclasses import dataclass
from typing import Optional, Callable, List
import numpy as np

from app.backend.config.config_manager import ConfigManager
from app.backend.pipeline.pipeline import PipelineStage


@dataclass
class MissionStatus:
    current_state: PipelineStage = PipelineStage.IDLE
    is_running: bool = False
    error_message: Optional[str] = None
    mission_time: float = 0.0

class DroneModel:
    def __init__(self):
        self.status = MissionStatus()
        self._frame: Optional[np.ndarray] = None
        self._state_changed_callbacks: List[Callable[[PipelineStage], None]] = []
        self._frame_updated_callbacks: List[Callable[[np.ndarray], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []
        
    def update_state(self, new_state: PipelineStage) -> None:
        """Update mission state and notify observers."""
        self.status.current_state = new_state
        for callback in self._state_changed_callbacks:
            callback(new_state)
            
    def update_frame(self, frame: np.ndarray) -> None:
        """Update current frame and notify observers."""
        self._frame = frame
        
        # Save frame if debug mode is enabled
        if ConfigManager().is_debug_mode:
            import cv2
            import os
            debug_path = ConfigManager().tello_config.debug_output_path
            frame_path = os.path.join(debug_path, f"frame_{self.status.mission_time:.2f}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        for callback in self._frame_updated_callbacks:
            callback(frame)
            
    def report_error(self, error: str) -> None:
        """Report error and notify observers."""
        self.status.error_message = error
        for callback in self._error_callbacks:
            callback(error)
            
    def update_mission_time(self, time: float) -> None:
        """Update mission time."""
        self.status.mission_time = time

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        return self._frame
