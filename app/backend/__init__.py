"""Backend package for the drone facial recognition system.

This package contains:
- Pipeline: Core processing pipeline and nodes
- Devices: Hardware abstraction layer
- Config: Configuration management
- Mission Manager: Overall mission control
"""

from .config.config_manager import ConfigManager
from .devices.tello import TelloFactory
from .mission_manager import MissionManager
from .pipeline.idle_node import Idle
from .pipeline.launch_node import Launch
from .pipeline.pipeline import Pipeline, PipelineNodeType

# from .pipeline.nodes import  ScanNode, IdentifyNode, TrackNode, ReturnNode

__all__ = [
    'ConfigManager',
    'TelloFactory',
    'MissionManager',
    'Pipeline',
    'PipelineNodeType',
    'Idle',
    'Launch',
    # 'ScanNode',
    # 'IdentifyNode',
    # 'TrackNode',
    # 'ReturnNode'
]
