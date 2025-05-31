import logging
from typing import Dict, Any, Optional
from ..devices.tello import TelloFactory, TelloDevice
from ..container import DroneData
from ... import logger


class DroneController:
    """Manages drone interactions and telemetry."""

    def __init__(self):
        self.tello = TelloFactory.create_tello()

    def connect(self) -> bool:
        """Connect to the Tello drone."""
        try:
            logger.info("Connecting to Tello device...")
            if not self.tello.is_connected:
                raise RuntimeError("Failed to connect to Tello device")
            logger.info("Tello device connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Tello: {str(e)}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the drone."""
        if self.tello:
            try:
                if self.tello.is_flying:
                    self.tello.land()
                self.tello.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting drone: {str(e)}")

    def stop_movement(self) -> None:
        """Stop drone movement."""
        if self.tello and self.tello.is_flying:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                logger.info("Drone movement stopped")
            except Exception as e:
                logger.error(f"Failed to stop drone movement: {str(e)}")

    def land(self) -> None:
        """Land the drone."""
        if self.tello and self.tello.is_flying:
            try:
                self.tello.land()
                logger.info("Drone landing initiated")
            except Exception as e:
                logger.error(f"Failed to land drone: {str(e)}")

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        if self.tello:
            try:
                self.tello.emergency()
                logger.critical("Drone emergency stop executed")
            except Exception as e:
                logger.error(f"Failed to execute drone emergency stop: {str(e)}")

    def get_telemetry(self) -> Optional[DroneData]:
        """Retrieve drone telemetry data."""
        if not self.tello:
            return None
        try:
            telemetry_data = {
                'height': self.tello.get_height(),
                'battery': self.tello.get_battery(),
                'temperature': self.tello.get_temperature(),
                'flight_time': self.tello.get_flight_time() if hasattr(self.tello, 'get_flight_time') else 0.0,
            }
            try:
                telemetry_data.update({
                    'speed_x': self.tello.get_speed_x() if hasattr(self.tello, 'get_speed_x') else 0.0,
                    'speed_y': self.tello.get_speed_y() if hasattr(self.tello, 'get_speed_y') else 0.0,
                    'speed_z': self.tello.get_speed_z() if hasattr(self.tello, 'get_speed_z') else 0.0,
                    'acceleration_x': self.tello.get_acceleration_x() if hasattr(self.tello, 'get_acceleration_x') else 0.0,
                    'acceleration_y': self.tello.get_acceleration_y() if hasattr(self.tello, 'get_acceleration_y') else 0.0,
                    'acceleration_z': self.tello.get_acceleration_z() if hasattr(self.tello, 'get_acceleration_z') else 0.0,
                })
            except:
                pass
            return DroneData(**telemetry_data)
        except Exception as e:
            logger.error(f"Telemetry update failed: {str(e)}")
            return None

    @property
    def is_connected(self) -> bool:
        """Check if drone is connected."""
        return self.tello.is_connected if self.tello else False