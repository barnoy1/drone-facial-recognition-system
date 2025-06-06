import numpy as np
from typing import Dict, Any

from ..container import MissionStatus, MissionState
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineNodeType
from app.core.face.inference import process_image  # Import face detection function
from ... import logger


class Emergency(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        self.node = PipelineNodeType.EMERGENCY_STOP
        self.name = PipelineNodeType.EMERGENCY_STOP.value

    def process(self, mission_state: MissionState, nodes: Dict, current_node: PipelineNode) -> PipelineNode:
        try:
            if self.is_done():
                from app.backend import Launch
                current_node = nodes.get(PipelineNodeType.IDLE)
                mission_state.status = MissionStatus.READY
                current_node.state = PipelineState.PENDING
                return current_node
            else:
                if not self.tello.is_connected or mission_state.frame_data is None:
                    self.state = PipelineState.FAILED
                    mission_state.status = MissionStatus.ERROR
                else:
                    self.state = PipelineState.FAILED
                    mission_state.status = MissionStatus.EMERGENCY_STOPPED
            return current_node
        except Exception as e:
            logger.error(f'an error has occurred in node :\n{e}')
            raise

    def reset(self) -> None:
        pass
