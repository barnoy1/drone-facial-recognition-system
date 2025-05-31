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

