from PySide6.QtCore import QTimer
from typing import Callable, Dict, Any, Optional
import numpy as np

from .pipeline.pipeline import Pipeline, PipelineState
from .devices.tello import TelloFactory, TelloDevice
from .pipeline.nodes import LaunchNode, ScanNode, IdentifyNode, TrackNode, ReturnNode
from .config.config_manager import ConfigManager

class MissionManager:
    """Manages the overall mission, including pipeline and device coordination."""
    
    def __init__(self):
        self.tello: Optional[TelloDevice] = None
        self.pipeline: Optional[Pipeline] = None
        self.mission_time_elapsed = 0
        self.mission_time_limit = 300  # 5 minutes
        
        # Timers
        self.mission_timer = QTimer()
        self.mission_timer.timeout.connect(self._update_mission_time)
        self.mission_timer.setInterval(1000)  # 1 second
        
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self._process_frame)
        self.stream_timer.setInterval(33)  # ~30 FPS
        
        # Status flags
        self.is_running = False
        self.error_state = False
        
    def initialize(self) -> bool:
        """Initialize mission manager."""
        try:
            # Create Tello device
            self.tello = TelloFactory.create_tello()
            if not self.tello.is_connected:
                raise RuntimeError("Failed to connect to Tello device")
                
            # Create and configure pipeline
            self.pipeline = Pipeline()
            self._setup_pipeline()
            
            return True
            
        except Exception as e:
            self.error_state = True
            self._handle_error(f"Initialization failed: {str(e)}")
            return False
            
    def _setup_pipeline(self) -> None:
        """Set up pipeline nodes."""
        if not self.pipeline or not self.tello:
            raise RuntimeError("Pipeline or Tello device not initialized")
            
        config = ConfigManager()
        pipeline_config = config.config.get('pipeline', {})
        
        # Create and register pipeline nodes
        self.pipeline.register_node(PipelineState.LAUNCH, LaunchNode(self.tello))
        self.pipeline.register_node(PipelineState.SCAN, 
                                  ScanNode(pipeline_config.get('scan', {}).get('scan_interval', 1.0)))
        self.pipeline.register_node(PipelineState.IDENTIFY, 
                                  IdentifyNode(pipeline_config.get('identify', {}).get('confidence_threshold', 0.85)))
        self.pipeline.register_node(PipelineState.TRACK, 
                                  TrackNode(pipeline_config.get('track', {}).get('update_rate', 0.1)))
        self.pipeline.register_node(PipelineState.RETURN, ReturnNode(self.tello))
        
    def start_mission(self) -> None:
        """Start the mission."""
        if self.is_running:
            return
            
        self.is_running = True
        self.mission_timer.start()
        self.stream_timer.start()
        
    def stop_mission(self) -> None:
        """Stop the mission normally."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.mission_timer.stop()
        self.stream_timer.stop()
        
        if self.tello:
            self.tello.land()
            
    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        self.is_running = False
        self.mission_timer.stop()
        self.stream_timer.stop()
        
        if self.tello:
            self.tello.emergency()
            
    def _process_frame(self) -> None:
        """Process the next frame through the pipeline."""
        if not self.is_running or not self.tello or not self.pipeline:
            return
            
        try:
            frame = self.tello.get_frame()
            if frame is None:
                raise RuntimeError("Failed to get frame from device")
                
            # Process frame through pipeline
            result = self.pipeline.process_frame(frame)
            if result:
                print(f"Pipeline update: {result}")
                
            # Check for completion or errors
            if self.pipeline.state == PipelineState.COMPLETE:
                self.stop_mission()
            elif self.pipeline.state == PipelineState.ERROR:
                self._handle_error(self.pipeline.error or "Unknown pipeline error")
                
        except Exception as e:
            self._handle_error(f"Frame processing error: {str(e)}")
            
    def _update_mission_time(self) -> None:
        """Update mission time and check limits."""
        self.mission_time_elapsed += 1
        if self.mission_time_elapsed >= self.mission_time_limit:
            print("Mission time limit reached")
            self.stop_mission()
            
    def _handle_error(self, error: str) -> None:
        """Handle error conditions."""
        self.error_state = True
        print(f"Error: {error}")
        self.emergency_stop()
