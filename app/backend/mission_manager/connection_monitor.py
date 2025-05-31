import time
import logging
from typing import Optional
from .drone_controller import DroneController
from .mission_config import MissionConfig
from .callback_manager import CallbackManager
from ..container import MissionState, MissionStatus
logger = logging.getLogger("app_logger")

class ConnectionMonitor:
    """Monitors drone connection health and telemetry."""

    def __init__(self, drone_controller: DroneController, config: MissionConfig, callback_manager: CallbackManager):
        self.drone_controller = drone_controller
        self.config = config
        self.callback_manager = callback_manager
        self.connection_check_failures = 0

    def check_connection(self, mission_state: MissionState) -> None:
        """Monitor drone connection health."""
        try:
            is_connected = self.drone_controller.is_connected
            if not is_connected:
                self.connection_check_failures += 1
                logger.warning(f"Connection check failed ({self.connection_check_failures}/{self.config.max_connection_failures})")
                if self.connection_check_failures >= self.config.max_connection_failures:
                    mission_state.connection_lost = True
                    self.callback_manager.notify_error("Lost connection to drone")
            else:
                if self.connection_check_failures > 0:
                    logger.info("Connection restored")
                self.connection_check_failures = 0
                mission_state.connection_lost = False
            self.callback_manager.notify_state_update(mission_state)
        except Exception as e:
            logger.error(f"Connection check error: {str(e)}")
            self.connection_check_failures += 1

    def update_telemetry(self, mission_state: MissionState) -> None:
        """Update drone telemetry data."""
        telemetry_data = self.drone_controller.get_telemetry()
        if telemetry_data:
            mission_state.drone_data = telemetry_data
            mission_state.last_telemetry_update = time.time()
            battery_level = telemetry_data.battery
            if battery_level <= self.config.battery_critical_threshold:
                if not mission_state.battery_critical:
                    logger.warning(f"Battery level critical: {battery_level}%")
                    mission_state.battery_critical = True
            else:
                mission_state.battery_critical = False
        else:
            mission_state.connection_lost = True
            self.check_connection_health(mission_state)

    def check_connection_health(self, mission_state: MissionState) -> None:
        """Check overall connection health."""
        current_time = time.time()
        if (mission_state.last_telemetry_update > 0 and
                current_time - mission_state.last_telemetry_update > self.config.connection_timeout):
            logger.warning("Telemetry data is stale - possible connection issue")
            mission_state.connection_lost = True

    def reset(self) -> None:
        """Reset connection monitor state."""
        self.connection_check_failures = 0