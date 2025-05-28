from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from enum import Enum, auto

from app import logger
from app.backend.container import PipelineStage, PipelineState
from app.backend.devices.tello import TelloDevice
from app.backend.mission_manager import MissionState


class PipelineNode(ABC):

    def __init__(self, tello: TelloDevice):
        self.tello = tello
        self.state = PipelineState.PENDING

    """Abstract base class for pipeline nodes."""
    @abstractmethod
    def process(self, mission_state: MissionState, context: Dict[str, Any]) -> bool:
        """Process a frame and update context. Return True when node is complete."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset node state."""
        pass

    def is_done(self) -> bool:
        return self.state in (PipelineStage.COMPLETE, PipelineState.ERROR)

class Pipeline:
    """Main pipeline manager."""
    
    def __init__(self):
        self.nodes: Dict[PipelineStage, PipelineNode] = {}
        self.current_state = PipelineStage.IDLE
        self.context: Dict[str, Any] = {}
        self.state_transitions = {
            PipelineStage.IDLE: PipelineStage.LAUNCH,
            PipelineStage.LAUNCH: PipelineStage.SCAN,
            PipelineStage.SCAN: PipelineStage.IDENTIFY,
            PipelineStage.IDENTIFY: PipelineStage.TRACK,
            PipelineStage.TRACK: PipelineStage.RETURN,
            PipelineStage.RETURN: PipelineStage.COMPLETE,
        }
    
    def register_node(self, state: PipelineStage, node: PipelineNode) -> None:
        """Register a node for a specific pipeline state."""
        self.nodes[state] = node
    
    def process_frame(self, mission_state: MissionState) -> Optional[str]:
        """Process a frame through the current pipeline node."""

        current_node = self.nodes.get(self.current_state)
        if not current_node:
            self.current_state = PipelineState.ERROR
            return f"No node registered for state: {self.current_state}"

        try:
            current_node.process(mission_state, self.context)
            if current_node.is_done():
                # Node is complete, transition to next state
                next_state = self.state_transitions.get(self.current_state)
                if next_state:
                    self.current_state = next_state
                    return f"Transitioned to state: {next_state}"

        except Exception as e:
            self.current_state = PipelineState.ERROR
            logger.error(f"Error in {self.current_state} node: {str(e)}")
            raise
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.current_state = PipelineStage.IDLE
        self.context.clear()
        for node in self.nodes.values():
            node.reset()
            
    @property
    def state(self) -> PipelineStage:
        return self.current_state
