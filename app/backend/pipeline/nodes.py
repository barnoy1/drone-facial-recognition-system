import numpy as np
from typing import Dict, Any

from ..container import MissionStatus
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineStage
from app.core.face.inference import process_image  # Import face detection function
from ..mission_manager import MissionState
from ... import logger


# We'll use this from the existing core module

class IdleNode(PipelineNode):
    def __init__(self, tello: TelloDevice):
        self.tello = tello
        self.state = PipelineState.PENDING
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        if not context.get('launched', False):
            success = self.tello.takeoff()
            context['launched'] = success
            return success
        return True

    def reset(self) -> None:
        pass


class IdleNode(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)

    def process(self, mission_state: MissionState, context: Dict[str, Any]) -> bool:
        try:

            if not self.is_done():
                if self.tello.is_connected and mission_state.frame_data is not None:
                    if MissionStatus.READY:
                        self.state = PipelineState.IN_PROGRESS
                    elif MissionStatus.RUNNING:
                        self.state = PipelineState.COMPLETE
                    else:
                        self.state = PipelineState.ERROR
                else:
                    self.state = PipelineState.ERROR
                    return False
        except Exception as e:
            logger.error(f'an error has occurred in node [IDLE]:\n{e}')
            raise

    def reset(self) -> None:
        pass

class LaunchNode(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        if not context.get('launched', False):
            success = self.tello.takeoff()
            context['launched'] = success
            return success
        return True
        
    def reset(self) -> None:
        pass

class ScanNode(PipelineNode):
    def __init__(self, scan_interval: float = 1.0):
        self.scan_interval = scan_interval
        self.last_scan_time = 0
        
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        current_time = context.get('time', 0)
        if current_time - self.last_scan_time >= self.scan_interval:
            # Perform face detection on the frame
            faces = process_image(frame)
            if faces:
                context['detected_faces'] = faces
                return True
            self.last_scan_time = current_time
        return False
        
    def reset(self) -> None:
        self.last_scan_time = 0

class IdentifyNode(PipelineNode):
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        faces = context.get('detected_faces', [])
        if not faces:
            return False
            
        # TODO: Implement face recognition using core.face.inference
        # For now, we'll just simulate identification
        context['identified_face'] = faces[0]
        return True
        
    def reset(self) -> None:
        pass

class TrackNode(PipelineNode):
    def __init__(self, update_rate: float = 0.1):
        self.update_rate = update_rate
        self.last_update_time = 0
        
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        current_time = context.get('time', 0)
        if current_time - self.last_update_time >= self.update_rate:
            face = context.get('identified_face')
            if face is None:
                return True  # Lost track, move to return
                
            # Update tracking
            # TODO: Implement actual tracking logic
            self.last_update_time = current_time
            
        return False
        
    def reset(self) -> None:
        self.last_update_time = 0

class ReturnNode(PipelineNode):
    def __init__(self, tello: TelloDevice):
        self.tello = tello
        self.landing_initiated = False
        
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        if not self.landing_initiated:
            success = self.tello.land()
            self.landing_initiated = success
            return success
        return True
        
    def reset(self) -> None:
        self.landing_initiated = False
