from typing import Optional, Callable, List
import numpy as np

from app import logger
from app.backend import ConfigManager, MissionManager
from app.backend.container import PipelineState

from app.backend.devices.tello import TelloFactory
from app.backend.mission_manager import MissionState
from app.backend.pipeline.nodes import LaunchNode, ScanNode, IdentifyNode, TrackNode, ReturnNode
from app.backend.pipeline.pipeline import Pipeline, PipelineStage
from app.frontend.callbacks import DroneModel
from app.frontend.app_view import AppView


class Presenter:
    def __init__(self, args, view: AppView):

        self.mission_manager = MissionManager(args=args,
                                              cb_on_state_changed=self._on_state_changed,
                                              cb_on_frame_updated=self._on_frame_updated,
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
        if not self.pipeline:
            self.view.log_message("System not initialized")
            return
        self.view.set_mission_running(True)
        self.view.log_message("Mission started")

        
    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        if self.tello:
            self.tello.emergency()
        self.model.status.is_running = False
        self.view.set_mission_running(False)
        self.view.log_message("EMERGENCY STOP EXECUTED")
        
    def process_frame(self, frame: np.ndarray) -> None:
        """Process a new frame through the pipeline."""
        if not self.model.status.is_running or not self.pipeline:
            return
            
        # Update model with new frame
        self.model.update_frame(frame)
        
        # Process frame through pipeline
        result = self.pipeline.process_frame(frame)
        if result:
            self.view.log_message(result)
            
        # Update state
        self._update_pipeline_state(self.pipeline.state)
        
    def _update_pipeline_state(self, state: PipelineStage) -> None:
        """Update pipeline state in model and view."""
        self.mission_manager.update_state(state)
        self.view.update_pipeline_state(state)
        
    def _on_state_changed(self, state: PipelineStage) -> None:
        """Handle state changes."""
        if state == PipelineStage.END_MISSION:
            self.model.status.is_running = False
            self.view.set_mission_running(False)
            self.view.log_message("Mission completed successfully")
        elif state == PipelineState.ERROR:
            self.emergency_stop()
            
    def _on_frame_updated(self, frame: np.ndarray) -> None:
        """Handle frame updates."""
        self.view.update_frame(frame)
        
    def _on_error(self, error: str) -> None:
        """Handle errors."""
        self.view.log_message(f"Error: {error}")
        self.emergency_stop()
