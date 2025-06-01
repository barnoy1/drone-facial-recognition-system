import cv2
import numpy as np
import os
import time
import argparse
import logging
import warnings
from pathlib import Path
from app.core.face.utils.create_embedding import create_stable_features, save_features
from app.core.face.utils.face_utils import process_image
from app.core.face.utils.visualization import draw_results

# Configure logging and suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger('app_logger')

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir(project_root):
    """Get the data directory."""
    return project_root / 'data'


def try_camera(device_id, width=640, height=480, fps=30):
    """Try to open and configure a camera device."""
    logger.debug(f"Attempting to open camera {device_id}...")

    device_path = f"/dev/video{device_id}"
    if not os.path.exists(device_path):
        logger.debug(f"Camera device {device_path} does not exist")
        return None

    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.debug(f"Could not open camera {device_id}")
        return None

    # Configure camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.debug(f"Could not read frame from camera {device_id}")
        cap.release()
        return None

    logger.debug(f"Successfully opened camera {device_id}")
    logger.debug(f"Frame shape: {frame.shape}")
    return cap


def do_inference(args_dict):
    """Perform real-time face recognition with provided arguments."""
    # Extract arguments from dictionary
    project_root = args_dict['project_root']
    embeddings_file = args_dict['embeddings_file']
    reference_dir = args_dict['reference_dir']
    max_device_id = args_dict['max_device_id']
    camera_width = args_dict['camera_width']
    camera_height = args_dict['camera_height']
    camera_fps = args_dict['camera_fps']
    fps_report_interval = args_dict['fps_report_interval']
    window_title = args_dict['window_title']
    input_frame = args_dict.get('input_frame')  # Optional input frame
    display_frame = args_dict.get('display_frame')  # Optional display frame
    external_trigger = args_dict.get('external_trigger') # Optional display frame result using opencv
    # Load or create features
    try:
        reference_features = np.load(embeddings_file, allow_pickle=True).item()
    except FileNotFoundError:
        logger.debug("No existing features found. Creating new reference features...")
        name, features = create_stable_features(reference_dir)
        save_features(name, features, embeddings_file.parent)
        reference_features = np.load(embeddings_file, allow_pickle=True).item()

    # Try to open the camera if no input frame is provided
    cap = None
    if input_frame is None:
        for device_id in range(max_device_id):
            cap = try_camera(device_id, camera_width, camera_height, camera_fps)
            if cap is not None:
                break

        if cap is None:
            logger.debug("Error: Could not open any camera")
            return

        # Get actual camera properties
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.debug(f"Camera properties: {actual_width}x{actual_height} @ {actual_fps}fps")
    else:
        logger.debug("Using provided input frame for processing")

    logger.debug("Starting video capture. Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()

    while True:
        # Capture frame-by-frame if no input frame is provided
        if input_frame is None:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.debug("Error: Failed to read frame")
                break
        else:
            frame = input_frame
            # If input_frame is provided, process only once
            input_frame = None  # Clear to prevent infinite loop

        # Use provided display frame or create a shallow copy of the input frame
        display = display_frame if display_frame is not None else np.copy(frame)

        results = process_image(frame, reference_features)

        # Draw results on display frame
        draw_results(display, results)

        if external_trigger:
            return display, results

        if fps_report_interval:
            frame_count += 1
            if frame_count % fps_report_interval == 0:  # Print FPS at specified interval
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                logger.debug(f"FPS: {fps:.2f}")

        # Display the resulting frame
        cv2.imshow(window_title, display)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final statistics
    if fps_report_interval:
        if frame_count > 0:
            total_time = time.time() - start_time
            average_fps = frame_count / total_time
            logger.debug(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds")
            logger.debug(f"Average FPS: {average_fps:.2f}")

    # Release the webcam if used
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function to collect arguments and call do_inference."""
    parser = argparse.ArgumentParser(description='Real-time Face Recognition System')
    parser.add_argument('--embeddings-file', type=str, help='File to store/load face features')
    parser.add_argument('--reference-dir', type=str, default=None, help='Directory containing reference face images')
    parser.add_argument('--max-device-id', type=int, default=2, help='Maximum camera device ID to try')
    parser.add_argument('--camera-width', type=int, default=640, help='Camera frame width')
    parser.add_argument('--camera-height', type=int, default=480, help='Camera frame height')
    parser.add_argument('--camera-fps', type=int, default=30, help='Camera frames per second')
    parser.add_argument('--fps-report-interval', type=int, default=30, help='Interval (frames) for FPS reporting')
    parser.add_argument('--window-title', type=str, default='Facial Recognition', help='Window title for display')
    parser.add_argument('--input-frame', type=str, default=None, help='Path to input frame (NumPy array) file')
    parser.add_argument('--display-frame', type=str, default=None, help='Path to display frame (NumPy array) file')
    parser.add_argument('--external_trigger', action='store_true', default=None, help='display result (NumPy array) file')

    args = parser.parse_args()

    # Setup paths
    project_root = get_project_root()
    data_dir = get_data_dir(project_root)

    # Use provided reference_dir or default to lfw_dataset/target
    reference_dir = Path(args.reference_dir) if args.reference_dir else data_dir / "lfw_dataset/target"

    # Use provided embeddings_file or default
    embeddings_file = Path(args.embeddings_file) if args.embeddings_file else (data_dir / "embeddings.npy")

    # Load input_frame and display_frame if provided
    input_frame = np.load(args.input_frame, allow_pickle=True) if args.input_frame else None
    display_frame = np.load(args.display_frame, allow_pickle=True) if args.display_frame else None

    args_dict = dict(
        project_root=project_root,
        embeddings_file=embeddings_file,
        reference_dir=reference_dir,
        max_device_id=args.max_device_id,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        fps_report_interval=args.fps_report_interval,
        window_title=args.window_title,
        input_frame=input_frame,
        display_frame=display_frame,
        external_trigger=False
    )

    do_inference(args_dict)


if __name__ == "__main__":
    main()