from PySide6.QtCore import QTimer
from .mission_config import MissionConfig
from .frame_processor import FrameProcessor
from .telemetry_manager import TelemetryManager
from ..container import MissionState

class TimerManager:
    """Manages timers for frame processing and connection monitoring."""

    def __init__(self, config: MissionConfig, frame_processor: FrameProcessor, telemetry_manager: TelemetryManager):
        self.config = config
        self.frame_processor = frame_processor
        self.telemetry_manager = telemetry_manager
        self.stream_timer = QTimer()
        self.connection_timer = QTimer()
        self._setup_timers()

    def _setup_timers(self) -> None:
        """Set up mission timers."""
        self.stream_timer.timeout.connect(lambda: self.frame_processor.process_frame())
        self.stream_timer.setInterval(int(1000 / self.config.frame_rate))
        self.connection_timer.timeout.connect(lambda: self.telemetry_manager.check_connection())
        self.connection_timer.setInterval(2000)  # 2 seconds

    def start_timers(self) -> None:
        """Start all timers."""
        if not self.stream_timer.isActive():
            self.stream_timer.start()
        if not self.connection_timer.isActive():
            self.connection_timer.start()

    def stop_stream_timer(self) -> None:
        """Stop the stream timer."""
        if self.stream_timer.isActive():
            self.stream_timer.stop()

    def stop_all_timers(self) -> None:
        """Stop all timers."""
        if self.stream_timer.isActive():
            self.stream_timer.stop()
        if self.connection_timer.isActive():
            self.connection_timer.stop()