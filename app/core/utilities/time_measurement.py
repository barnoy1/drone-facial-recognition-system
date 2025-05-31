import time

from app.backend.mission_manager.camera_manager import logger


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds to run")
        return result

    return wrapper


def calc_duration(start, end):
    from datetime import datetime

    if start is None or end is None:
        return '00:00:00'

    # Calculate the duration
    delta = start - end

    # Extract total seconds and milliseconds
    total_seconds = int(delta.total_seconds())
    milliseconds = int(delta.microseconds / 1000)

    # Compute hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format the duration as HH:MM:SS.ms
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return duration_str