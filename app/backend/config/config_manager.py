
import os
import yaml
from dataclasses import dataclass
from typing import Optional

from app import logger
from app.core.utilities.json import pretty_print_dict


@dataclass
class PipelineConfig:
    skip_launch_node: bool = False
    skip_find_target_node: bool = False
    skip_detect_face_node: bool = False
    skip_identify_face_node: bool = False
    skip_track_target_node: bool = False
    skip_find_home_node: bool = False
    skip_land_node: bool = False


@dataclass
class TelloConfig:
    mock_enabled: bool = False
    mock_source: str = ""  # "webcam" or "folder"
    mock_folder_path: Optional[str] = None
    debug_mode: bool = False
    debug_output_path: str = ""
    battery_critical_threshold: float = 0.0


class ConfigManager:
    config = None
    output_dir = None
    args = None
    tello_config = TelloConfig()
    _initialized = False

    @staticmethod
    def initialize(args: str) -> None:
        ConfigManager.args = args
        ConfigManager.output_dir = args.output_dir
        if ConfigManager._initialized:
            print("ConfigManager already initialized")
            return
        ConfigManager.load_config(ConfigManager.args.config)
        ConfigManager._initialized = True

    @staticmethod
    def load_config(config_path: str) -> None:
        """Load configuration from YAML file."""

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            ConfigManager.config = yaml.safe_load(f)
        logger.info(f"Loading config from: {config_path}")

        # Parse Tello configuration
        tello_config = ConfigManager.config.get('tello', {})
        pipeline_config = ConfigManager.config.get('pipeline', {})
        logger.info(pretty_print_dict('Tello config from file', tello_config))

        ConfigManager.tello_config = TelloConfig(
            mock_enabled=tello_config.get('mock_enabled', False),
            mock_source=tello_config.get('mock_source', 'webcam'),
            mock_folder_path=tello_config.get('mock_folder_path'),
            debug_mode=tello_config.get('debug_mode', False),
            battery_critical_threshold=tello_config.get('battery_critical_threshold', '0.25'),
            debug_output_path=tello_config.get('rel_debug_output_path', 'debug/frames')
        )

        ConfigManager.pipeline_config = PipelineConfig(
            skip_launch_node=pipeline_config.get('skip_launch_node', False),
            skip_find_target_node=pipeline_config.get('skip_find_target_node', False),
            skip_detect_face_node=pipeline_config.get('skip_detect_face_node', False),
            skip_identify_face_node=pipeline_config.get('skip_identify_face_node', False),
            skip_track_target_node=pipeline_config.get('skip_track_target_node', False),
            skip_find_home_node=pipeline_config.get('skip_find_home_node', False),
            skip_land_node=pipeline_config.get('skip_land_node', False)
        )

        ConfigManager.tello_config.debug_output_path = os.path.join(ConfigManager.output_dir,
                                                                    ConfigManager.tello_config.debug_output_path)

        os.makedirs(ConfigManager.tello_config.debug_output_path, exist_ok=True)

    @staticmethod
    def is_mock_enabled() -> bool:
        if not ConfigManager._initialized:
            raise RuntimeError("ConfigManager not initialized. Call initialize() first.")
        return ConfigManager.tello_config.mock_enabled

    @staticmethod
    def is_debug_mode() -> bool:
        if not ConfigManager._initialized:
            raise RuntimeError("ConfigManager not initialized. Call initialize() first.")
        return ConfigManager.tello_config.debug_mode
