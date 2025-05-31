from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from enum import Enum, auto

from app import logger
from app.backend.container import PipelineNodeType, PipelineState
from app.backend.devices.tello import TelloDevice
from app.backend.mission_manager import MissionState


class PipelineNode(ABC):

    def __init__(self, tello: TelloDevice):
        self.tello = tello
        self.state = PipelineState.PENDING

    """Abstract base class for pipeline nodes."""

    @abstractmethod
    def process(self, mission_state: MissionState, nodes: Dict, next_node: object) -> PipelineNodeType:
        """Process a frame and update context. Return True when node is complete."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset node state."""
        pass

    def is_done(self) -> bool:
        return self.state in (PipelineState.COMPLETED, PipelineState.FAILED)


class PipelineStage:
    pass


class Pipeline:
    """Main pipeline manager."""

    def __init__(self):
        self.nodes: Dict[PipelineNodeType, PipelineNode] = {}
        self.current_node = None
        self.context: Dict[str, Any] = {}

    def register_node(self, node: PipelineNodeType, node_class: PipelineNode) -> None:
        """Register a node for a specific pipeline state."""
        self.nodes[node] = node_class

    def process_frame(self, mission_state: MissionState) -> bool:
        """Process a frame through the current pipeline node."""

        if not self.current_node:
            self.current_node = PipelineStage
            return f"No node registered for state: {self.current_node}"

        try:
            next_node = self.current_node.process(mission_state=mission_state,
                                                  current_node=self.current_node,
                                                  nodes=self.nodes)

            if self.current_node.is_done() and next_node is not None:
                # Node is complete, transition to next state
                if self.current_node != next_node:
                    logger.info(f"Transitioned from: [{self.current_node.name}] to [{next_node.name}]")
                    # Update mission state
                    mission_state.pipeline_previous_node = self.current_node
                    mission_state.pipeline_current_node = next_node

                    # Update mission state with pipeline state
                    self.current_node = next_node

                    return True

            return False

        except Exception as e:
            self.current_node = PipelineState.FAILED
            logger.error(f"Error in {self.current_node} node: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset pipeline state."""
        self.current_node = PipelineNodeType.IDLE
        self.context.clear()
        for node in self.nodes.values():
            node.reset()

    @property
    def get_current_node(self) -> PipelineNodeType:
        return self.current_node
