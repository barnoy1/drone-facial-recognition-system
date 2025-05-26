from typing import Optional
import numpy as np

from app.backend import ConfigManager
from app.backend.devices.tello import TelloFactory
from app.backend.pipeline.nodes import LaunchNode, ScanNode, IdentifyNode, TrackNode, ReturnNode
from app.backend.pipeline.pipeline import Pipeline, PipelineState
from app.frontend.model.drone_model import DroneModel
from app.frontend.view.drone_view import DroneView


class DronePresenter:
    def __init__(self, model: DroneModel, view: DroneView):
        self.model = model
        self.view = view
        self.pipeline: Optional[Pipeline] = None
        self.tello = None

        # Connect view signals
        self.view.start_mission.connect(self.start_mission)
        self.view.emergency_stop.connect(self.emergency_stop)
        
        # Connect model callbacks
        self.model.register_state_callback(self._on_state_changed)
        self.model.register_frame_callback(self._on_frame_updated)
        self.model.register_error_callback(self._on_error)
        
    def initialize(self) -> bool:
        """Initialize the system."""
        try:




            # Create Tello device
            self.tello = TelloFactory.create_tello()
            if not self.tello.connect():
                self.view.log_message("Failed to connect to Tello device")
                return False
                
            # Initialize pipeline
            self.pipeline = Pipeline()
            
            # Register pipeline nodes
            self.pipeline.register_node(PipelineState.LAUNCH, LaunchNode(self.tello))
            self.pipeline.register_node(PipelineState.SCAN, ScanNode())
            self.pipeline.register_node(PipelineState.IDENTIFY, IdentifyNode())
            self.pipeline.register_node(PipelineState.TRACK, TrackNode())
            self.pipeline.register_node(PipelineState.RETURN, ReturnNode(self.tello))
            
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
            
        self.model.status.is_running = True
        self.view.set_mission_running(True)
        self.view.log_message("Mission started")
        self._update_pipeline_state(PipelineState.LAUNCH)
        
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
        
    def _update_pipeline_state(self, state: PipelineState) -> None:
        """Update pipeline state in model and view."""
        self.model.update_state(state)
        self.view.update_pipeline_state(state)
        
    def _on_state_changed(self, state: PipelineState) -> None:
        """Handle state changes."""
        if state == PipelineState.COMPLETE:
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
