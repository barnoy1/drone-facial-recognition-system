from time import sleep

import numpy as np
from typing import Dict, Any

from .. import ConfigManager
from ..container import MissionStatus, MissionState
from ..devices.tello import TelloDevice
from .pipeline import PipelineNode, PipelineState, PipelineNodeType
from app.core.face.inference import process_image  # Import face detection function
from ... import logger

class IdentifyFace(PipelineNode):
    def __init__(self, tello: TelloDevice):
        super().__init__(tello)
        self.node = PipelineNodeType.IDENTIFY_FACE
        self.name = PipelineNodeType.IDENTIFY_FACE.value
        self.face_config = ConfigManager.face_config
    def process(self, mission_state: MissionState, nodes: Dict, current_node: PipelineNode) -> PipelineNode:
        try:
            if self.is_done():
                from app.backend import Track
                current_node = nodes.get(PipelineNodeType.TRACK_TARGET)
                mission_state.status = MissionStatus.RUNNING
                return current_node
            else:
                self.state = PipelineState.IN_PROGRESS
                if not self.tello.is_connected or mission_state.frame_data is None:
                    self.state = PipelineState.FAILED
                    mission_state.status = MissionStatus.ERROR

                # debugging (skip this state)
                if ConfigManager.pipeline_config.skip_identify_face_node:
                    current_node = nodes.get(PipelineNodeType.TRACK_TARGET)
                    self.state = PipelineState.SKIPPED
                    sleep(1)
                    return current_node

                # Load input_frame and display_frame if provided
                input_frame = mission_state.frame_data.raw_frame
                display_frame = mission_state.frame_data.display_frame

                w, h = input_frame.shape[:-1]
                args_dict = dict(
                    project_root=None,
                    embeddings_file=ConfigManager.face_config.embeddings_file,
                    reference_dir=None,
                    max_device_id=None,
                    camera_width=w,
                    camera_height=h,
                    camera_fps=None,
                    fps_report_interval=None,
                    window_title=None,
                    input_frame=input_frame,
                    display_frame=display_frame,
                    external_trigger=True
                )

                from app.core.face.inference import do_inference
                (mission_state.frame_data.display_frame,
                 mission_state.detected_faces) \
                    = do_inference(args_dict)

                if mission_state.detected_faces:
                    logger.info("Detected faces detected.")
                    for i, candidate in enumerate(mission_state.detected_faces):
                        if candidate.get('similarity') >= self.face_config.similarity_threshold:
                            current_node = nodes.get(PipelineNodeType.TRACK_TARGET)
                            self.state = PipelineState.COMPLETED
                        return current_node

            return current_node
        except Exception as e:
            logger.error(f'an error has occurred in node :\n{e}')
            raise

    def reset(self) -> None:
        pass
