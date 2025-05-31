import time
import logging
import numpy as np
from typing import Optional

from .callback_manager import CallbackManager
from .drone_controller import DroneController
from app.backend.mission_manager.camera_manager import CameraManager
from ..pipeline.pipeline import Pipeline, PipelineNodeType
from ..pipeline.emergency_node import Emergency
from ..container import MissionState, PipelineState

logger = logging.getLogger("app_logger")

class FrameProcessor:
    """Manages frame processing and pipeline execution."""

    def __init__(self, drone_controller: DroneController, pipeline: Pipeline,
                 camera_manager: CameraManager, callback_manager: CallbackManager):
        self.drone_controller = drone_controller
        self.camera_manager = camera_manager
        self.pipeline = pipeline
        self.callback_manager = callback_manager
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.mission_state = MissionState()

    def initialize(self) -> None:
        """Initialize frame processor components."""
        logger.info("Initializing frame processor...")
        self.camera_manager = self.camera_manager
        self.pipeline = self.pipeline

        logger.info("Frame processor initialized")

    def process_frame(self, mission_state: MissionState) -> bool:
        """Process the next frame through the pipeline."""
        if not self.camera_manager or not self.pipeline:
            return False

        try:
            frame_data = self.camera_manager.get_frame()
            if frame_data is None:
                return False

            mission_state.frame_data = frame_data
            mission_state.drone_data = self.drone_controller.get_telemetry()
            self._update_frame_stats(mission_state)
            mission_state.state_has_changed_trigger = self.pipeline.process_frame(mission_state)
            self.camera_manager.add_overlay(mission_state)
            self.callback_manager.notify_frame_update(np.array(mission_state.frame_data.display_frame))
            if mission_state.state_has_changed_trigger:
                self.callback_manager.notify_state_update(mission_state)
            return True

        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return False

    def _update_frame_stats(self, mission_state: MissionState) -> None:
        """Update frame processing statistics."""
        mission_state.frames_processed += 1
        current_time = time.time()
        if self.last_frame_time > 0:
            frame_interval = current_time - self.last_frame_time
            if frame_interval > 0:
                instant_fps = 1.0 / frame_interval
                mission_state.fps = np.round((mission_state.fps * 0.9) + (instant_fps * 0.1), 2)
        self.last_frame_time = current_time
        logger.debug(f"Processing at {mission_state.fps:.1f} FPS, total frames: {mission_state.frames_processed}")

    def reset(self) -> None:
        """Reset frame processor state."""
        self.frame_count = 0
        self.last_frame_time = 0.0
        if self.pipeline:
            self.pipeline.reset()

    def reset_pipeline(self) -> None:
        """Reset pipeline to initial state."""
        if self.pipeline:
            self.pipeline.current_node = self.pipeline.nodes[PipelineNodeType.IDLE]
            self.pipeline.current_node.state = PipelineState.COMPLETED
            self.mission_state.pipeline_current_node = self.pipeline.current_node

    def cleanup(self) -> None:
        """Clean up frame processor resources."""
        if self.camera_manager:
            try:
                self.camera_manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up camera manager: {str(e)}")

    def set_pipeline_to_idle(self) -> None:
        """Set pipeline to IDLE state with COMPLETED status."""
        if self.pipeline:
            self.pipeline.current_node = self.pipeline.nodes[PipelineNodeType.IDLE]
            self.pipeline.current_node.state = PipelineState.COMPLETED
            self.mission_state.pipeline_current_node = self.pipeline.current_node
            logger.info("Pipeline set to IDLE state with COMPLETED status")

    def set_pipeline_to_emergency(self) -> None:
        """Set pipeline to EMERGENCY_STOP state with FAILED status."""
        if self.pipeline:
            self.pipeline.current_node = self.pipeline.nodes[PipelineNodeType.EMERGENCY_STOP]
            self.pipeline.current_node.state = PipelineState.FAILED
            self.mission_state.pipeline_current_node = self.pipeline.current_node
            self.mission_state.external_trigger = True
            logger.info("Pipeline set to EMERGENCY_STOP state with FAILED status")