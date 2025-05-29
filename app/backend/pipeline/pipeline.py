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
        return self.state in (PipelineNodeType.END_MISSION, PipelineState.ERROR)


class Pipeline:
    """Main pipeline manager."""

    def __init__(self):
        self.nodes: Dict[PipelineNodeType, PipelineNode] = {}
        self.next_node = None
        self.current_node = PipelineNodeType.IDLE
        self.context: Dict[str, Any] = {}

    def register_node(self, node: PipelineNodeType, node_class: PipelineNode) -> None:
        """Register a node for a specific pipeline state."""
        self.nodes[node] = node_class

    def process_frame(self, mission_state: MissionState) -> Optional[str]:
        """Process a frame through the current pipeline node."""

        self.current_node = self.nodes.get(self.current_node)
        if not self.current_node:
            self.current_node = PipelineState.ERROR
            return f"No node registered for state: {self.current_node}"

        try:
            self.current_node.process(mission_state=mission_state,
                                      nodes=self.nodes,
                                      next_node=self.next_node)
            if self.current_node.is_done() and self.next_node != self.current_node:
                # Node is complete, transition to next state
                if self.current_node != self.next_node:
                    return f"Transitioned to state: {self.next_node}"

        except Exception as e:
            self.current_node = PipelineState.ERROR
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
