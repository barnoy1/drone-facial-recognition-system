import numpy as np
from typing import Dict, Any

from ..container import MissionStatus
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineNodeType
from app.core.face.inference import process_image  # Import face detection function
from ..mission_manager import MissionState
from ... import logger

class LaunchNode(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        self.node = PipelineNodeType.LAUNCH
        self.name = __class__.__name__

    def process(self, mission_state: MissionState, context: Dict[str, Any]) -> bool:
        try:

            if not self.is_done():
                if self.tello.is_connected and mission_state.frame_data is not None:
                    if MissionStatus.READY:
                        self.state = PipelineState.IN_PROGRESS
                    elif MissionStatus.RUNNING:
                        self.state = PipelineState.COMPLETED
                        # self.next_state =
                    else:
                        self.state = PipelineState.FAILED
                else:
                    self.state = PipelineState.FAILED
                    return False
        except Exception as e:
            logger.error(f'an error has occurred in node [IDLE]:\n{e}')
            raise

    def reset(self) -> None:
        pass
