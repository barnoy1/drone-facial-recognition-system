from ..config.config_manager import ConfigManager

class MissionConfig:
    """Manages mission configuration parameters."""

    def __init__(self, args: object):
        """Initialize with configuration from args."""
        ConfigManager.initialize(args)
        self._load_mission_config()

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
        self.max_connection_failures = 3
        self.fps_update_interval = 5.0  # seconds