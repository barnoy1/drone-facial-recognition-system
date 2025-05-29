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
        self.name = __class__.__name__
    def process(self, mission_state: MissionState, nodes: Dict, current_node: PipelineNode) -> PipelineNode:
        try:
            self.state = PipelineState.IN_PROGRESS

            if not self.is_done():
                if not self.tello.is_connected or mission_state.frame_data is None:
                    self.state = PipelineState.FAILED
            elif MissionStatus.RUNNING:
                self.state = PipelineState.COMPLETED
                from app.backend import LaunchNode
                current_node = nodes.get(PipelineNodeType.LAUNCH)
            return current_node
        except Exception as e:
            logger.error(f'an error has occurred in node [IDLE]:\n{e}')
            raise

    def reset(self) -> None:
        pass
