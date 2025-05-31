from typing import Optional, Callable, List
import numpy as np

from app import logger
from app.backend import MissionManager
from app.backend.container import MissionState, PipelineNodeType
from app.frontend.app_view import AppView

class Presenter:
    """Coordinates between the view and the mission manager."""

    def __init__(self, args, view: AppView):
        self.view = view
        self.mission_manager = MissionManager.instance()
        self.mission_manager.initialize(
            args=args,
            cb_on_state_changed=self._on_state_changed,
            cb_on_update_pipeline_state=self._on_state_changed,  # Use same callback for pipeline state
            cb_on_frame_updated=self._on_frame_updated,
            cb_on_error=self._on_error
        )


        # Connect view signals
        self.view.start_mission.connect(self.start_mission)
        self.view.emergency_stop.connect(self.emergency_stop)

    def initialize(self) -> bool:
        """Initialize the system."""
        try:
            if not self.mission_manager.drone_controller.is_connected:
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
        self.mission_manager.frame_processor.set_pipeline_to_idle()
        self.mission_manager.start_mission()
        self.view.set_mission_running(True)
        self.view.log_message("Mission started")

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        self.mission_manager.frame_processor.set_pipeline_to_emergency()
        self.mission_manager.emergency_stop()
        self.view.set_mission_running(False)
        self.view.log_message("EMERGENCY STOP EXECUTED")

    def _on_state_changed(self, mission_state: MissionState) -> None:
        """Handle state changes."""
        self.view.update_pipeline_state(mission_state)
        if mission_state.state_has_changed_trigger:
            self.view.log_message(
                f"state changed from "
                f"[{mission_state.pipeline_previous_node.name}] to "
                f"[{mission_state.pipeline_current_node.name}]"
            )

    def _on_frame_updated(self, frame: np.ndarray) -> None:
        """Handle frame updates."""
        self.view.update_frame(frame)

    def _on_error(self, error: str) -> None:
        """Handle errors."""
        self.view.log_message(f"Error: {error}")
        self.emergency_stop()