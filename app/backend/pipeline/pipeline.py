from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from enum import Enum, auto

class PipelineState(Enum):
    IDLE = auto()
    LAUNCH = auto()
    SCAN = auto()
    IDENTIFY = auto()
    TRACK = auto()
    RETURN = auto()
    COMPLETE = auto()
    ERROR = auto()

class PipelineNode(ABC):
    """Abstract base class for pipeline nodes."""
    
    @abstractmethod
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> bool:
        """Process a frame and update context. Return True when node is complete."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset node state."""
        pass

class Pipeline:
    """Main pipeline manager."""
    
    def __init__(self):
        self.nodes: Dict[PipelineState, PipelineNode] = {}
        self.current_state = PipelineState.IDLE
        self.context: Dict[str, Any] = {}
        self.state_transitions = {
            PipelineState.IDLE: PipelineState.LAUNCH,
            PipelineState.LAUNCH: PipelineState.SCAN,
            PipelineState.SCAN: PipelineState.IDENTIFY,
            PipelineState.IDENTIFY: PipelineState.TRACK,
            PipelineState.TRACK: PipelineState.RETURN,
            PipelineState.RETURN: PipelineState.COMPLETE,
        }
    
    def register_node(self, state: PipelineState, node: PipelineNode) -> None:
        """Register a node for a specific pipeline state."""
        self.nodes[state] = node
    
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Process a frame through the current pipeline node."""
        if self.current_state in (PipelineState.COMPLETE, PipelineState.ERROR):
            return None
            
        current_node = self.nodes.get(self.current_state)
        if not current_node:
            self.current_state = PipelineState.ERROR
            return f"No node registered for state: {self.current_state}"
            
        try:
            # Process frame in current node
            if current_node.process(frame, self.context):
                # Node is complete, transition to next state
                next_state = self.state_transitions.get(self.current_state)
                if next_state:
                    self.current_state = next_state
                    return f"Transitioned to state: {next_state}"
        except Exception as e:
            self.current_state = PipelineState.ERROR
            return f"Error in {self.current_state} node: {str(e)}"
            
        return None
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.current_state = PipelineState.IDLE
        self.context.clear()
        for node in self.nodes.values():
            node.reset()
            
    @property
    def state(self) -> PipelineState:
        return self.current_state
