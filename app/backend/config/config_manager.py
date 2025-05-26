import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class TelloConfig:
    mock_enabled: bool = False
    mock_source: str = ""  # "webcam" or "folder"
    mock_folder_path: Optional[str] = None
    debug_mode: bool = False
    debug_output_path: str = ""


class ConfigManager:
    config = None
    tello_config = TelloConfig()
    _initialized = False

    @staticmethod
    def initialize(config_path: str) -> None:
        """
        Initialize the ConfigManager by loading the configuration file.
        This method ensures the config is loaded only once.
        """
        if ConfigManager._initialized:
            print("ConfigManager already initialized")
            return
        ConfigManager.load_config(config_path)
        ConfigManager._initialized = True

    @staticmethod
    def load_config(config_path: str) -> None:
        """Load configuration from YAML file."""
        print(f"Loading config from: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            ConfigManager.config = yaml.safe_load(f)

        print(f"Loaded config: {ConfigManager.config}")

        # Parse Tello configuration
        tello_config = ConfigManager.config.get('tello', {})
        print(f"Tello config from file: {tello_config}")

        ConfigManager.tello_config = TelloConfig(
            mock_enabled=tello_config.get('mock_enabled', False),
            mock_source=tello_config.get('mock_source', 'webcam'),
            mock_folder_path=tello_config.get('mock_folder_path'),
            debug_mode=tello_config.get('debug_mode', False),
            debug_output_path=tello_config.get('debug_output_path', 'debug_frames')
        )

        print(f"Final tello config: mock_enabled={ConfigManager.tello_config.mock_enabled}, "
              f"mock_source={ConfigManager.tello_config.mock_source}")

        if ConfigManager.tello_config.debug_mode and not os.path.exists(ConfigManager.tello_config.debug_output_path):
            os.makedirs(ConfigManager.tello_config.debug_output_path)

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
