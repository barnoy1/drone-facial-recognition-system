import logging
import os
import colorlog
import sys
from datetime import datetime
import logging
import os

logger = logging.getLogger("app_logger")

# Check if the logger has handlers, to prevent reconfiguring
if not logger.hasHandlers():
    # Generate the filename with current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs('out/logs', exist_ok=True)
    log_filename = f'out/logs/app_{current_time}.log'
    logger.setLevel(logging.INFO)

    # Create a file handler for logging to a file (plain text)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Define a formatter for the file (plain text, no colors)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s => %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Create a stream handler for logging to the console (with colors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define a color formatter for the console
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s - %(message)s",  # Color for level name
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Attach the color formatter to the console handler
    console_handler.setFormatter(color_formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Log an example message
    logger.info(f"logger created in current working directory: {os.getcwd()}")
