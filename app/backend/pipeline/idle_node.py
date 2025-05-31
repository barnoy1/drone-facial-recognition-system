import numpy as np
from typing import Dict, Any

from ..container import MissionStatus
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineNodeType
from app.core.face.inference import process_image  # Import face detection function
from ..mission_manager import MissionState
from ... import logger


class Idle(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        self.node = PipelineNodeType.IDLE
        self.name = __class__.__name__

    def process(self, mission_state: MissionState, nodes: Dict, current_node: PipelineNode) -> PipelineNode:
        try:
            if self.is_done():
                from app.backend import Launch
                current_node = nodes.get(PipelineNodeType.LAUNCH)
                mission_state.status = MissionStatus.RUNNING
                return current_node
            else:
                if not self.tello.is_connected or mission_state.frame_data is None:
                    self.state = PipelineState.FAILED
                    mission_state.status = MissionStatus.ERROR
            return current_node
        except Exception as e:
            logger.error(f'an error has occurred in node [IDLE]:\n{e}')
            raise

    def reset(self) -> None:
        pass
