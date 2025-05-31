from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import logger


@dataclass
class HUD:
    FONT = "arial.ttf"  # Path to font file (update to valid path or use default)
    FONT_SCALE = 20  # Font size in pixels (adjusted for PIL, ~0.7 in OpenCV)
    FONT_COLOR = (255, 255, 255)  # White text (RGB)
    THICKNESS = 2  # Not used in PIL; kept for compatibility
    LINE_TYPE = None  # Not used in PIL; kept for compatibility
    PADDING = 10
    LINE_SPACING = 30
    Y_OFFSET = 80

    def __post_init__(self):
        """Initialize font after dataclass creation."""
        try:
            self.font = ImageFont.truetype(self.FONT, size=self.FONT_SCALE)
            print(f"Loaded font: {self.FONT}")
        except IOError as e:
            logger.warning(f"Warning: Could not load font '{self.FONT}': {e}. Using default font.")
            try:
                # Try an alternative system font
                self.font = ImageFont.truetype("dejavu/DejaVuSans.ttf", size=self.FONT_SCALE)
                print("Loaded fallback font: dejavu/DejaVuSans.ttf")
            except IOError:
                # Fallback to PIL's default font
                self.font = ImageFont.load_default()
                print("Using PIL default font")
                # Note: load_default() may not support size in older Pillow versions
                if not hasattr(self.font, 'getmask'):
                    raise RuntimeError("Default font is invalid; ensure Pillow is up-to-date")
@dataclass
class FrameData:
    """Container for frame data and metadata."""
    raw_frame: Optional[np.ndarray] = None          # Original frame for processing
    display_frame: Optional[np.ndarray] = None      # Frame with overlays for display
    timestamp: float = 0.0
    resolution: Tuple[int, int] = (960, 720)
    frame_number: int = 0



class PipelineState(Enum):
    SKIPPED = auto()
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()


class PipelineNodeType(Enum):
    EMERGENCY_STOP = auto()
    IDLE = auto()
    LAUNCH = auto()
    SCAN = auto()
    IDENTIFY = auto()
    TRACK = auto()
    RETURN = auto()
    END_MISSION = auto()


class MissionStatus(Enum):
    """Mission status enumeration."""
    NOT_INITIALIZED = "not_initialized"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    EMERGENCY_STOPPED = "emergency_stopped"


@dataclass
class DroneData:
    """Data retrieved from drone device."""
    height: float = 0.0
    battery: int = 0
    temperature: float = 0.0
    flight_time: float = 0.0
    speed_x: float = 0.0
    speed_y: float = 0.0
    speed_z: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    acceleration_z: float = 0.0


@dataclass
class MissionState:
    """Current state of the mission."""
    status: MissionStatus = MissionStatus.NOT_INITIALIZED
    pipeline_current_node: PipelineNodeType = None
    pipeline_previous_node: PipelineNodeType = None
    drone_data: Optional[DroneData] = None
    detected_faces: List[Dict[str, Any]] = field(default_factory=list)
    mission_time: float = 0.0
    error: Optional[str] = None
    is_running: bool = False
    is_paused: bool = False
    frame_data: Optional[FrameData] = None
    battery_critical: bool = False
    connection_lost: bool = False
    initialization_complete: bool = False
    last_telemetry_update: float = 0.0
    frames_processed: int = 0
    fps: float = 0.0
    state_has_changed_trigger: bool = False
