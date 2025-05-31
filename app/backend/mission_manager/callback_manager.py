import logging
from typing import Callable, List
import numpy as np
from ..container import MissionState, PipelineNodeType
logger = logging.getLogger("app_logger")

class CallbackManager:
    """Manages registration and notification of callbacks."""

    def __init__(self):
        self._state_changed_callbacks: List[Callable[[PipelineNodeType], None]] = []
        self._frame_updated_callbacks: List[Callable[[np.ndarray], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []

    def register_state_callback(self, callback: Callable[[PipelineNodeType], None]) -> None:
        """Register callback for state updates."""
        self._state_changed_callbacks.append(callback)
        logger.debug(f"State callback registered. Total callbacks: {len(self._state_changed_callbacks)}")

    def register_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register callback for frame updates."""
        self._frame_updated_callbacks.append(callback)
        logger.debug(f"Frame callback registered. Total callbacks: {len(self._frame_updated_callbacks)}")

    def register_error_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for error notifications."""
        self._error_callbacks.append(callback)
        logger.debug(f"Error callback registered. Total callbacks: {len(self._error_callbacks)}")

    def notify_state_update(self, mission_state: MissionState) -> None:
        """Notify all registered state callbacks."""
        try:
            for callback in self._state_changed_callbacks:
                try:
                    callback(mission_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {str(e)}")
        except Exception as e:
            logger.error(f"Error notifying state update: {str(e)}")

    def notify_frame_update(self, frame: np.ndarray) -> None:
        """Notify all registered frame callbacks."""
        try:
            for callback in self._frame_updated_callbacks:
                try:
                    callback(frame)
                except Exception as e:
                    logger.error(f"Error in frame callback: {str(e)}")
        except Exception as e:
            logger.error(f"Error notifying frame update: {str(e)}")

    def notify_error(self, error: str) -> None:
        """Notify all registered error callbacks."""
        try:
            for callback in self._error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error in error callback: {str(e)}")
        except Exception as e:
            logger.error(f"Error notifying error: {str(e)}")