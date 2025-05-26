import os
import yaml
from dataclasses import dataclass
from typing import Optional

from app.core.common.singleton import Singleton


@dataclass
class TelloConfig:
    mock_enabled: bool = False
    mock_source: str = ""  # "webcam" or "folder"
    mock_folder_path: Optional[str] = None
    debug_mode: bool = False
    debug_output_path: str = ""


@Singleton
class ConfigManager:
    def __init__(self):
        self.config = None
        self.tello_config = TelloConfig()
        self._initialized = False

    def initialize(self, config_path: str) -> None:
        """
        Initialize the ConfigManager by loading the configuration file.
        This method ensures the config is loaded only once.
        """
        if self._initialized:
            print("ConfigManager already initialized")
            return
        self.load_config(config_path)
        self._initialized = True

    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        print(f"Loading config from: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"Loaded config: {self.config}")

        # Parse Tello configuration
        tello_config = self.config.get('tello', {})
        print(f"Tello config from file: {tello_config}")

        self.tello_config = TelloConfig(
            mock_enabled=tello_config.get('mock_enabled', False),
            mock_source=tello_config.get('mock_source', 'webcam'),
            mock_folder_path=tello_config.get('mock_folder_path'),
            debug_mode=tello_config.get('debug_mode', False),
            debug_output_path=tello_config.get('debug_output_path', 'debug_frames')
        )

        print(f"Final tello config: mock_enabled={self.tello_config.mock_enabled}, "
              f"mock_source={self.tello_config.mock_source}")

        if self.tello_config.debug_mode and not os.path.exists(self.tello_config.debug_output_path):
            os.makedirs(self.tello_config.debug_output_path)

    @property
    def is_mock_enabled(self) -> bool:
        if not self._initialized:
            raise RuntimeError("ConfigManager not initialized. Call initialize() first.")
        return self.tello_config.mock_enabled

    @property
    def is_debug_mode(self) -> bool:
        if not self._initialized:
            raise RuntimeError("ConfigManager not initialized. Call initialize() first.")
        return self.tello_config.debug_mode