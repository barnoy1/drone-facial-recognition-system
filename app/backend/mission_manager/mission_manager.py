import numpy as np
from PySide6.QtCore import QObject
from typing import Callable, Dict, Any
from app.backend.mission_manager.mission_config import MissionConfig
from .camera_manager import  CameraManager
from .drone_controller import DroneController
from .frame_processor import FrameProcessor
from .connection_monitor import ConnectionMonitor
from .callback_manager import CallbackManager
from .timer_manager import TimerManager
from app.core.utilities.decorators.singleton import Singleton
from .. pipeline.pipeline import Pipeline
from ..container import MissionState, MissionStatus, PipelineNodeType, DroneData
from app import logger
from ..pipeline.idle_node import Idle
from ..pipeline.launch_node import Launch
from ..pipeline.emergency_node import Emergency

@Singleton
class MissionManager(QObject):
    """Central orchestrator for the drone mission.

    Responsibilities:
    - Mission lifecycle management (start, pause, resume, stop, emergency stop).
    - Coordinates between components (drone, frame processing, connection monitoring).
    """

    def __init__(self):
        super().__init__()
        self.mission_state = MissionState()
        self.config = None
        self.drone_controller = None
        self.frame_processor = None
        self.connection_monitor = None
        self.callback_manager = None
        self.timer_manager = None
        self.pipeline = None

    def initialize(self, args: object,
                   cb_on_state_changed: Callable[[PipelineNodeType], None],
                   cb_on_update_pipeline_state: Callable[[PipelineNodeType], None],
                   cb_on_frame_updated: Callable[[np.ndarray], None],
                   cb_on_telemetry_updated: Callable[[MissionState], None],
                   cb_on_error: Callable[[str], None]) -> None:
        """Initialize the mission manager and its components."""
        logger.info("MissionManager initializing...")
        self.config = MissionConfig(args)
        self.mission_state.status = MissionStatus.NOT_INITIALIZED
        self.callback_manager = CallbackManager()
        self.callback_manager.register_state_callback(cb_on_state_changed)
        self.callback_manager.register_state_callback(cb_on_update_pipeline_state)
        self.callback_manager.register_frame_callback(cb_on_frame_updated)
        self.callback_manager.register_telemetry_callback(cb_on_telemetry_updated)
        self.callback_manager.register_error_callback(cb_on_error)

        self.drone_controller = DroneController()
        self.camera_manager = CameraManager(self.drone_controller.tello)
        self.pipeline = Pipeline()
        self._setup_pipeline()

        self.frame_processor = FrameProcessor(self.drone_controller, self.pipeline, self.camera_manager, self.callback_manager)
        self.connection_monitor = ConnectionMonitor(self.drone_controller, self.config, self.callback_manager)
        self.timer_manager = TimerManager(self.config, self.frame_processor, self.connection_monitor)

        if self._initialize_internal():
            self.mission_state.status = MissionStatus.READY
            self.mission_state.initialization_complete = True
        self.callback_manager.notify_state_update(self.mission_state)
        logger.info("MissionManager initialized")

    def _initialize_internal(self) -> bool:
        """Initialize internal components."""
        try:
            self.drone_controller.connect()
            self.frame_processor.initialize()
            self.timer_manager.start_timers()
            return True
        except Exception as e:
            self.callback_manager.notify_error(f"Initialization failed: {str(e)}")
            return False

    def start_mission(self) -> None:
        """Start the mission."""
        if self.mission_state.status not in [MissionStatus.READY, MissionStatus.PAUSED]:
            logger.warning(f"Cannot start mission from status: {self.mission_state.status}")
            return
        if not self.mission_state.initialization_complete:
            logger.error("Cannot start mission - initialization not complete")
            return

        logger.info("Starting mission")
        self.mission_state.status = MissionStatus.RUNNING
        self.mission_state.is_running = True
        self.mission_state.is_paused = False
        self.mission_state.error = None
        self.mission_state.mission_time = 0.0
        self.mission_state.frames_processed = 0
        self.frame_processor.reset_pipeline()
        self.timer_manager.start_timers()
        self.callback_manager.notify_state_update(self.mission_state)
        logger.info("Mission started successfully")

    def stop_mission(self) -> None:
        """Stop the mission normally."""
        if self.mission_state.status in [MissionStatus.NOT_INITIALIZED, MissionStatus.READY]:
            return

        logger.info("Stopping mission")
        self.mission_state.status = MissionStatus.COMPLETED
        self.mission_state.is_running = False
        self.mission_state.is_paused = False
        self.timer_manager.stop_all_timers()
        self.drone_controller.land()
        self.callback_manager.notify_state_update(self.mission_state)
        logger.info("Mission stopped successfully")

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        logger.critical("Emergency stop initiated")
        self.mission_state.status = MissionStatus.EMERGENCY_STOPPED
        self.mission_state.is_running = False
        self.mission_state.is_paused = False
        self.mission_state.error = "Emergency stop executed"
        self.timer_manager.stop_all_timers()
        self.drone_controller.emergency_stop()
        self.callback_manager.notify_state_update(self.mission_state)


    def _setup_pipeline(self) -> None:
        """Set up the processing pipeline with all nodes."""
        if not self.pipeline or not self.drone_controller.tello:
            raise RuntimeError("Pipeline or Tello device not available")
        self.pipeline.register_node(PipelineNodeType.EMERGENCY_STOP, Emergency(self.drone_controller.tello))
        self.pipeline.register_node(PipelineNodeType.IDLE, Idle(self.drone_controller.tello))
        self.pipeline.register_node(PipelineNodeType.LAUNCH, Launch(self.drone_controller.tello))
        self.pipeline.current_node = self.pipeline.nodes[PipelineNodeType.IDLE]
        self.mission_state.pipeline_current_node = self.pipeline.current_node

        logger.info("Pipeline nodes registered successfully")

    @staticmethod
    def get_detailed_status(mission_state: MissionState) -> Dict[str, Any]:
        """Get comprehensive mission status information."""
        return {
            'status': mission_state.status.value,
            'pipeline_state': mission_state.pipeline_current_node.name,
            'is_running': mission_state.is_running,
            'is_paused': mission_state.is_paused,
            'mission_time': mission_state.mission_time,
            'frames_processed': mission_state.frames_processed,
            'fps': round(mission_state.fps, 1),
            'detected_faces_count': len(mission_state.detected_faces),
            'error': mission_state.error,
            'battery_critical': mission_state.battery_critical,
            'connection_lost': mission_state.connection_lost,
            'initialization_complete': mission_state.initialization_complete,
            'drone_data': mission_state.drone_data.__dict__ if mission_state.drone_data else None
        }