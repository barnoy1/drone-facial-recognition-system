from time import sleep

import numpy as np
from typing import Dict, Any

from .. import ConfigManager
from ..container import MissionStatus, MissionState
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineNodeType
from app.core.face.inference import process_image  # Import face detection function
from ... import logger

class Land(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        self.node = PipelineNodeType.LAND
        self.name = PipelineNodeType.LAND.value
    def process(self, mission_state: MissionState, nodes: Dict, current_node: PipelineNode) -> PipelineNode:
        try:
            if self.is_done():
                from app.backend import Idle
                # current_node = nodes.get(PipelineNodeType.Idle)
                mission_state.status = MissionStatus.SUCCESS
                return current_node
            else:
                self.state = PipelineState.IN_PROGRESS
                if not self.tello.is_connected or mission_state.frame_data is None:
                    self.state = PipelineState.FAILED
                    mission_state.status = MissionStatus.ERROR

                # debugging (skip this state)
                if ConfigManager.pipeline_config.skip_land_node:
                    # current_node = nodes.get(PipelineNodeType.IDLE)
                    self.state = PipelineState.SKIPPED
                    sleep(1)
                    return current_node
            return current_node
        except Exception as e:
            logger.error(f'an error has occurred in node :\n{e}')
            raise

    def reset(self) -> None:
        pass
