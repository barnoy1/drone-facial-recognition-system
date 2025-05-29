import numpy as np
from typing import Dict, Any

from ..container import MissionStatus
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineNodeType
from app.core.face.inference import process_image  # Import face detection function
from ..mission_manager import MissionState
from ... import logger

class IdleNode(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        self.node = PipelineNodeType.IDLE

    def process(self, mission_state: MissionState, nodes: Dict, next_node: PipelineNode) -> bool:
        try:

            if not self.is_done():
                if self.tello.is_connected and mission_state.frame_data is not None:
                    if MissionStatus.READY:
                        self.state = PipelineState.IN_PROGRESS
                    elif MissionStatus.RUNNING:
                        self.state = PipelineState.COMPLETE
                        from app.backend import LaunchNode
                        next_node = nodes[PipelineNodeType.LAUNCH]
                        return True

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
