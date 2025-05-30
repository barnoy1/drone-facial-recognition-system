from typing import Optional, Callable, List
import numpy as np

from app import logger
from app.backend import ConfigManager, MissionManager
from app.backend.container import PipelineState

from app.backend.devices.tello import TelloFactory
from app.backend.mission_manager import MissionState
from app.backend.pipeline.pipeline import Pipeline, PipelineNodeType

from app.frontend.app_view import AppView


class Presenter:
    def __init__(self, args, view: AppView):

        self.mission_manager = MissionManager.instance()

        self.mission_manager.initialize(args=args,
                                        cb_on_state_changed=self._on_state_changed,
                                        cb_on_frame_updated=self._on_frame_updated,
                                        cb_on_update_pipeline_state=self._update_pipeline_state,
                                        cb_on_error=self._on_error)

        self.view = view
        self.pipeline: Optional[Pipeline] = None

        # Connect view signals
        self.view.start_mission.connect(self.start_mission)
        self.view.emergency_stop.connect(self.emergency_stop)

    def initialize(self) -> bool:
        """Initialize the system."""
        try:

            # Create Tello device
            if not self.mission_manager.tello.is_connected:
                self.view.log_message("Failed to connect to Tello device")
                return False

            self.view.log_message("System initialized successfully")
            return True

        except Exception as e:
            self.view.log_message(f"Initialization failed: {str(e)}")
            return False

    def start_mission(self) -> None:
        """Start the drone mission."""
        if not self.view.pipeline_nodes:
            self.view.log_message("System not initialized")
            return
        self.view.set_mission_running(True)
        self.view.log_message("Mission started")

        pipeline = self.mission_manager.pipeline
        curr_node = self.mission_manager.mission_state.pipeline_current_node
        self.mission_manager.mission_state.pipeline_current_node.state = PipelineState.COMPLETED

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        self.view.set_mission_running(False)
        self.view.log_message("EMERGENCY STOP EXECUTED")

    def _update_pipeline_state(self, state: PipelineNodeType) -> None:
        """Update pipeline state in model and view."""
        self.view.update_pipeline_state(state)

    def _on_state_changed(self, state: PipelineNodeType) -> None:
        """Handle state changes."""
        if state == PipelineNodeType.END_MISSION:
            self.model.status.is_running = False
            self.view.set_mission_running(False)
            self.view.log_message("Mission completed successfully")
        elif state == PipelineState.FAILED:
            self.emergency_stop()

    def _on_frame_updated(self, frame: np.ndarray) -> None:
        """Handle frame updates."""
        self.view.update_frame(frame)

    def _on_error(self, error: str) -> None:
        """Handle errors."""
        self.view.log_message(f"Error: {error}")
        self.emergency_stop()
