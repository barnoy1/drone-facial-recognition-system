import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import colorlog
from PySide6.QtCore import QTimer, QObject
from typing import Callable, Dict, Any, Optional, List
import numpy as np

from .camera_manager import FrameData, CameraManager
from .container import MissionState, MissionStatus, DroneData, PipelineState
from .navigation_manager import NavigationManager
from .pipeline.pipeline import Pipeline, PipelineStage
from .devices.tello import TelloFactory, TelloDevice
from .pipeline.nodes import LaunchNode, ScanNode, IdentifyNode, TrackNode, ReturnNode, IdleNode
from .config.config_manager import ConfigManager
from .. import logger


class MissionManager(QObject):
    """Central orchestrator for the drone mission.

    Responsibilities:
    1. State machine management and mission orchestration
    2. Data coordination between CameraManager and Pipeline
    3. Drone control and safety management
    4. Mission status and telemetry updates
    5. Error handling and recovery
    6. Thread-safe operations with Qt signals/slots
    """

    # Qt signals for thread-safe communication
    from PyQt6.QtCore import pyqtSignal
    state_changed = pyqtSignal(object)  # MissionState
    error_occurred = pyqtSignal(str)
    mission_completed = pyqtSignal()
    telemetry_updated = pyqtSignal(object)  # DroneData

    def __init__(self, args: object,
                 cb_on_state_changed: PipelineStage,
                 cb_on_frame_updated: np.ndarray,
                 cb_on_error: str) -> object:
        super().__init__()

        # Core components
        ConfigManager.initialize(args)

        self.tello: Optional[TelloDevice] = None
        self.pipeline: Optional[Pipeline] = None
        self.camera_manager: Optional[CameraManager] = None
        self.nav_manager: Optional[NavigationManager] = None
        self.mission_state = MissionState()

        # Mission configuration
        self._load_mission_config()

        # Frame processing tracking
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._fps_update_interval = 5.0  # seconds

        # Initialize timers
        self._setup_timers()

        # Callback registry
        self._state_callbacks: List[Callable[[PipelineState], None]] = []
        self._frame_updated_callbacks: List[Callable[[np.ndarray], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []

        # Safety monitoring
        self._connection_check_failures = 0
        self._max_connection_failures = 3

        self.initialize()
        logger.info("MissionManager initialized")

        self.register_state_callback(cb_on_state_changed)
        self.register_frame_callback(cb_on_frame_updated)
        self.register_error_callback(cb_on_error)

    def register_state_callback(self, callback: Callable[[PipelineStage], None]) -> None:
        """Register callback for state updates."""
        self._state_changed_callbacks.append(callback)
        logger.debug(f"State callback registered. Total callbacks: {len(self._state_changed_callbacks)}")

    def register_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register callback for state updates."""
        self._frame_updated_callbacks.append(callback)
        logger.debug(f"State callback registered. Total callbacks: {len(self._frame_updated_callbacks)}")

    def register_error_callback(self, callback: Callable[[str], None]) -> None:
        self._error_callbacks.append(callback)
        logger.debug(f"Error callback registered. Total callbacks: {len(self._error_callbacks)}")

    def _load_mission_config(self) -> None:
        """Load mission configuration from ConfigManager."""
        config = ConfigManager().config
        mission_config = config.get('mission', {})

        self.mission_time_limit = mission_config.get('time_limit', 300)  # 5 minutes default
        self.telemetry_interval = mission_config.get('telemetry_interval', 1.0)  # 1 second
        self.frame_rate = mission_config.get('frame_rate', 30)  # 30 FPS
        self.battery_critical_threshold = mission_config.get('battery_critical', 10)
        self.battery_emergency_threshold = mission_config.get('battery_emergency', 5)
        self.connection_timeout = mission_config.get('connection_timeout', 5.0)

    def _setup_timers(self) -> None:
        """Set up mission timers."""

        # Frame processing timer
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._process_frame)
        self.stream_timer.setInterval(int(1000 / self.frame_rate))  # Based on configured FPS
        self.stream_timer.start()

        # Connection monitoring timer
        self.connection_timer = QTimer(self)
        self.connection_timer.timeout.connect(self._check_connection)
        self.connection_timer.setInterval(2000)  # 2 seconds
        self.connection_timer.start()


    def initialize(self) -> bool:
        """Initialize mission manager components."""
        try:
            logger.info("Initializing mission manager components...")

            # Update status
            self.mission_state.status = MissionStatus.NOT_INITIALIZED
            self._notify_state_update()

            # Create and connect to Tello device
            logger.info("Connecting to Tello device...")
            self.tello = TelloFactory.create_tello()
            if not self.tello.is_connected:
                raise RuntimeError("Failed to connect to Tello device")
            logger.info("Tello device connected successfully")

            # Initialize camera manager
            logger.info("Initializing camera manager...")
            self.camera_manager = CameraManager(self.tello)
            self.nav_manager = NavigationManager(self.tello)
            logger.info("Camera manager initialized")

            # Create and configure pipeline
            logger.info("Setting up processing pipeline...")
            self.pipeline = Pipeline()
            self._setup_pipeline()
            logger.info("Processing pipeline configured")

            # Start connection monitoring
            self.connection_timer.start()

            # Update state
            self.mission_state.status = MissionStatus.READY
            self.mission_state.initialization_complete = True
            self._notify_state_update()

            logger.info("Mission manager initialization successful")
            return True

        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            self._handle_error(error_msg)
            return False

    def _setup_pipeline(self) -> None:
        """Set up the processing pipeline with all nodes."""
        if not self.pipeline or not self.tello:
            raise RuntimeError("Pipeline or Tello device not available")

        # Register pipeline nodes
        self.pipeline.register_node(PipelineStage.IDLE, IdleNode(self.tello))
        self.pipeline.register_node(PipelineStage.LAUNCH, LaunchNode(self.tello))
        self.pipeline.register_node(PipelineStage.SCAN, ScanNode())
        self.pipeline.register_node(PipelineStage.IDENTIFY, IdentifyNode())
        self.pipeline.register_node(PipelineStage.TRACK, TrackNode())
        self.pipeline.register_node(PipelineStage.RETURN, ReturnNode(self.tello))

        logger.info("Pipeline nodes registered successfully")

    def register_state_callback(self, callback: Callable[[MissionState], None]) -> None:
        """Register callback for state updates."""
        self._state_callbacks.append(callback)
        logger.info(f"State callback registered. Total callbacks: {len(self._state_callbacks)}")

    def register_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register callback for state updates."""
        self._frame_updated_callbacks.append(callback)
        logger.info(f"State callback registered. Total callbacks: {len(self._frame_updated_callbacks)}")

    def register_error_callback(self, callback: Callable[[str], None]) -> None:
        self._error_callbacks.append(callback)
        logger.info(f"Error callback registered. Total callbacks: {len(self._error_callbacks)}")

    def _notify_state_update(self) -> None:
        """Notify all registered callbacks of state changes."""
        try:
            # Emit Qt signal
            # self.state_changed.emit(self.mission_state)

            # Call registered callbacks
            for callback in self._state_callbacks:
                try:
                    callback(self.mission_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {str(e)}")

        except Exception as e:
            logger.error(f"Error notifying state update: {str(e)}")

    def _notify_frame_update(self) -> None:
        """Notify all registered callbacks of state changes."""
        try:
            # Emit Qt signal
            # self.state_changed.emit(self.mission_state)

            # Call registered callbacks
            for callback in self._frame_updated_callbacks:
                try:
                    callback(np.array(self.mission_state.frame_data.display_frame))
                except Exception as e:
                    logger.error(f"Error in state callback: {str(e)}")

        except Exception as e:
            logger.error(f"Error notifying state update: {str(e)}")

    def start_mission(self) -> None:
        """Start the mission."""
        if self.mission_state.status not in [MissionStatus.READY, MissionStatus.PAUSED]:
            logger.warning(f"Cannot start mission from status: {self.mission_state.status}")
            return

        if not self.mission_state.initialization_complete:
            logger.error("Cannot start mission - initialization not complete")
            return

        logger.info("Starting mission")

        # Update state
        self.mission_state.status = MissionStatus.RUNNING
        self.mission_state.is_running = True
        self.mission_state.is_paused = False
        self.mission_state.error = None
        self.mission_state.mission_time = 0.0
        self.mission_state.frames_processed = 0

        # Start timers
        self.mission_timer.start()
        self.stream_timer.start()
        self.telemetry_timer.start()
        self.fps_timer.start()

        # Initialize pipeline state
        if self.pipeline:
            self.mission_state.pipeline_state = PipelineStage.LAUNCH

        self._notify_state_update()
        logger.info("Mission started successfully")

    def pause_mission(self) -> None:
        """Pause the current mission."""
        if self.mission_state.status != MissionStatus.RUNNING:
            logger.warning(f"Cannot pause mission from status: {self.mission_state.status}")
            return

        logger.info("Pausing mission")

        # Update state
        self.mission_state.status = MissionStatus.PAUSED
        self.mission_state.is_paused = True

        # Stop processing timers but keep telemetry and connection monitoring
        self.stream_timer.stop()
        self.mission_timer.stop()

        # Hover the drone if it's flying
        if self.tello and self.tello.is_flying:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)  # Stop movement
                logger.info("Drone movement stopped for pause")
            except Exception as e:
                logger.error(f"Failed to stop drone movement: {str(e)}")

        self._notify_state_update()
        logger.info("Mission paused")

    def resume_mission(self) -> None:
        """Resume a paused mission."""
        if self.mission_state.status != MissionStatus.PAUSED:
            logger.warning(f"Cannot resume mission from status: {self.mission_state.status}")
            return

        logger.info("Resuming mission")

        # Update state
        self.mission_state.status = MissionStatus.RUNNING
        self.mission_state.is_paused = False

        # Restart timers
        self.mission_timer.start()
        self.stream_timer.start()

        self._notify_state_update()
        logger.info("Mission resumed")

    def stop_mission(self) -> None:
        """Stop the mission normally."""
        if self.mission_state.status in [MissionStatus.NOT_INITIALIZED, MissionStatus.READY]:
            return

        logger.info("Stopping mission")

        # Update state
        self.mission_state.status = MissionStatus.COMPLETED
        self.mission_state.is_running = False
        self.mission_state.is_paused = False

        # Stop all timers
        self._stop_all_timers()

        # Land the drone safely
        if self.tello and self.tello.is_flying:
            try:
                self.tello.land()
                logger.info("Drone landing initiated")
            except Exception as e:
                logger.error(f"Failed to land drone: {str(e)}")

        # Emit completion signal
        self.mission_completed.emit()
        self._notify_state_update()
        logger.info("Mission stopped successfully")

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        logger.critical("Emergency stop initiated")

        # Update state immediately
        self.mission_state.status = MissionStatus.EMERGENCY_STOPPED
        self.mission_state.is_running = False
        self.mission_state.is_paused = False
        self.mission_state.error = "Emergency stop executed"

        # Stop all timers
        self._stop_all_timers()

        # Emergency stop the drone
        if self.tello:
            try:
                self.tello.emergency()
                logger.critical("Drone emergency stop executed")
            except Exception as e:
                logger.error(f"Failed to execute drone emergency stop: {str(e)}")

        # Emit error signal
        # self.error_occurred.emit("Emergency stop executed")
        self._notify_state_update()

    def _stop_all_timers(self) -> None:
        """Stop all running timers."""
        timers = [self.mission_timer, self.stream_timer, self.telemetry_timer, self.fps_timer]
        for timer in timers:
            if timer.isActive():
                timer.stop()

    def _process_frame(self) -> None:
        """Process the next frame through the pipeline."""
        if not self.camera_manager or not self.pipeline:
            return

        try:
            # Get frame from camera manager
            frame_data = self.camera_manager.get_frame()
            if frame_data is None:
                self._connection_check_failures += 1
                if self._connection_check_failures >= self._max_connection_failures:
                    raise RuntimeError("Failed to get frame from camera - connection lost")
                return

            # Reset connection failure counter on successful frame
            self._connection_check_failures = 0
            self.mission_state.connection_lost = False
            self.mission_state.frame_data = frame_data

            self.mission_state.drone_data = self.nav_manager.get_measurements()

            # Process frame through pipeline
            self.pipeline.process_frame(self.mission_state)

            # Update frame processing statistics
            self.mission_state.frames_processed += 1
            current_time = time.time()
            if self._last_frame_time > 0:
                frame_interval = current_time - self._last_frame_time
                if frame_interval > 0:
                    instant_fps = 1.0 / frame_interval
                    # Smooth FPS calculation
                    self.mission_state.fps = (self.mission_state.fps * 0.9) + (instant_fps * 0.1)
            self._last_frame_time = current_time

            # Add overlays based on pipeline state
            self.camera_manager.add_overlay(self.mission_state)

            # Update mission state with pipeline state
            self.mission_state.pipeline_current_node = self.pipeline.current_node

            if self.mission_state.pipeline_current_node.state == PipelineState.COMPLETE:
                self._notify_state_update()

            # Handle pipeline completion or errors
            if self.pipeline.state == PipelineStage.END_MISSION:
                logger.info("Pipeline completed successfully")
                self.stop_mission()

            self._notify_frame_update()


        except Exception as e:
            self._handle_error(f"Frame processing error: {str(e)}")

    def _update_telemetry(self) -> None:
        """Update drone telemetry data."""
        if not self.tello:
            return

        try:
            # Get comprehensive telemetry data
            telemetry_data = {
                'height': self.tello.get_height(),
                'battery': self.tello.get_battery(),
                'temperature': self.tello.get_temperature(),
            }

            # Get additional telemetry if available
            try:
                telemetry_data.update({
                    'speed_x': self.tello.get_speed_x() if hasattr(self.tello, 'get_speed_x') else 0.0,
                    'speed_y': self.tello.get_speed_y() if hasattr(self.tello, 'get_speed_y') else 0.0,
                    'speed_z': self.tello.get_speed_z() if hasattr(self.tello, 'get_speed_z') else 0.0,
                    'acceleration_x': self.tello.get_acceleration_x() if hasattr(self.tello,
                                                                                 'get_acceleration_x') else 0.0,
                    'acceleration_y': self.tello.get_acceleration_y() if hasattr(self.tello,
                                                                                 'get_acceleration_y') else 0.0,
                    'acceleration_z': self.tello.get_acceleration_z() if hasattr(self.tello,
                                                                                 'get_acceleration_z') else 0.0,
                })
            except:
                pass  # Extended telemetry not available

            self.mission_state.drone_data = DroneData(**telemetry_data)
            self.mission_state.last_telemetry_update = time.time()

            # Check battery levels
            battery_level = telemetry_data['battery']
            if battery_level <= self.battery_critical_threshold:
                if not self.mission_state.battery_critical:
                    logger.warning(f"Battery level critical: {battery_level}%")
                    self.mission_state.battery_critical = True
            else:
                self.mission_state.battery_critical = False



        except Exception as e:
            logger.error(f"Telemetry update failed: {str(e)}")
            self.mission_state.connection_lost = True
            self._check_connection_health()

    def _check_connection(self) -> None:
        """Monitor drone connection health."""
        if not self.tello:
            return

        try:
            # Try a simple command to check connection
            is_connected = self.tello.is_connected

            if not is_connected:
                self._connection_check_failures += 1
                logger.warning(
                    f"Connection check failed ({self._connection_check_failures}/{self._max_connection_failures})")

                if self._connection_check_failures >= self._max_connection_failures:
                    self.mission_state.connection_lost = True
                    self._handle_error("Lost connection to drone")
            else:
                if self._connection_check_failures > 0:
                    logger.info("Connection restored")
                self._connection_check_failures = 0
                self.mission_state.connection_lost = False

        except Exception as e:
            logger.error(f"Connection check error: {str(e)}")
            self._connection_check_failures += 1

    def _check_connection_health(self) -> None:
        """Check overall connection health and take action if needed."""
        current_time = time.time()

        # Check if telemetry is too old
        if (self.mission_state.last_telemetry_update > 0 and
                current_time - self.mission_state.last_telemetry_update > self.connection_timeout):
            logger.warning("Telemetry data is stale - possible connection issue")
            self.mission_state.connection_lost = True

    def _update_mission_time(self) -> None:
        """Update mission time and check limits."""
        self.mission_state.mission_time += 1

        # Check mission time limit
        if self.mission_state.mission_time >= self.mission_time_limit:
            logger.warning(f"Mission time limit reached ({self.mission_time_limit}s)")
            self.stop_mission()
            return

    def _calculate_fps(self) -> None:
        """Calculate and log FPS statistics."""
        if self.mission_state.frames_processed > 0:
            logger.debug(f"Processing at {self.mission_state.fps:.1f} FPS, "
                         f"total frames: {self.mission_state.frames_processed}")

    def _handle_error(self, error: str) -> None:
        """Handle error conditions."""
        logger.error(f"Mission error: {error}")
        self.mission_state.error = error
        self.mission_state.status = MissionStatus.ERROR

        # Emit error signal
        # self.error_occurred.emit(error)

        # Execute emergency stop
        self.emergency_stop()

    def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive mission status information."""
        return {
            'status': self.mission_state.status.value,
            'pipeline_state': self.mission_state.pipeline_state.name if self.mission_state.pipeline_state else 'UNKNOWN',
            'is_running': self.mission_state.is_running,
            'is_paused': self.mission_state.is_paused,
            'mission_time': self.mission_state.mission_time,
            'frames_processed': self.mission_state.frames_processed,
            'fps': round(self.mission_state.fps, 1),
            'detected_faces_count': len(self.mission_state.detected_faces),
            'error': self.mission_state.error,
            'battery_critical': self.mission_state.battery_critical,
            'connection_lost': self.mission_state.connection_lost,
            'initialization_complete': self.mission_state.initialization_complete,
            'drone_data': {
                'battery': self.mission_state.drone_data.battery if self.mission_state.drone_data else 0,
                'height': self.mission_state.drone_data.height if self.mission_state.drone_data else 0.0,
                'temperature': self.mission_state.drone_data.temperature if self.mission_state.drone_data else 0.0,
                'flight_time': self.mission_state.drone_data.flight_time if self.mission_state.drone_data else 0.0,
            } if self.mission_state.drone_data else None
        }

    def reset_mission(self) -> bool:
        """Reset mission to ready state."""
        if self.mission_state.status == MissionStatus.RUNNING:
            logger.warning("Cannot reset mission while running")
            return False

        logger.info("Resetting mission")

        # Stop all timers
        self._stop_all_timers()

        # Reset state
        self.mission_state = MissionState()
        self.mission_state.status = MissionStatus.READY if self.mission_state.initialization_complete else MissionStatus.NOT_INITIALIZED

        # Reset counters
        self._frame_count = 0
        self._connection_check_failures = 0
        self._last_frame_time = 0.0

        # Reset pipeline if available
        if self.pipeline:
            self.pipeline.reset()

        self._notify_state_update()
        logger.info("Mission reset complete")
        return True

    def cleanup(self) -> None:
        """Clean up resources and connections."""
        logger.info("Cleaning up mission manager resources")

        # Stop mission if running
        if self.mission_state.is_running:
            self.stop_mission()

        # Stop all timers
        self._stop_all_timers()

        # Cleanup camera manager
        if self.camera_manager:
            try:
                self.camera_manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up camera manager: {str(e)}")

        # Disconnect drone
        if self.tello:
            try:
                if self.tello.is_flying:
                    self.tello.land()
                self.tello.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting drone: {str(e)}")

        logger.info("Mission manager cleanup complete")
