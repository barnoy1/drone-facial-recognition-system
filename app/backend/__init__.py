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
from .pipeline.pipeline import Pipeline, PipelineNodeType
from .pipeline.nodes import LaunchNode, ScanNode, IdentifyNode, TrackNode, ReturnNode

__all__ = [
    'ConfigManager',
    'TelloFactory',
    'MissionManager',
    'Pipeline',
    'PipelineNodeType',
    'IdleNode',
    'LaunchNode',
    'ScanNode',
    'IdentifyNode',
    'TrackNode',
    'ReturnNode'
]
