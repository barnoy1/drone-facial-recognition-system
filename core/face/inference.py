import cv2
import numpy as np
import os
import time
import argparse
import logging
import warnings
from pathlib import Path
from core.face.utils.create_embedding import create_stable_features, save_features
from core.face.utils.face_utils import process_image
from core.face.utils.visualization import draw_results

# Configure logging and suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def get_data_dir():
    """Get the data directory."""
    return get_project_root() / 'data'

def try_camera(device_id):
    """Try to open and configure a camera device."""
    print(f"Attempting to open camera {device_id}...")
    
    device_path = f"/dev/video{device_id}"
    if not os.path.exists(device_path):
        print(f"Camera device {device_path} does not exist")
        return None
        
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Could not open camera {device_id}")
        return None
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Could not read frame from camera {device_id}")
        cap.release()
        return None
        
    print(f"Successfully opened camera {device_id}")
    print(f"Frame shape: {frame.shape}")
    return cap

def main():
    """Main function for real-time streaming face recognition."""
    parser = argparse.ArgumentParser(description='Real-time Face Recognition System')
    parser.add_argument('--embeddings-file', type=str,
                      help='File to store/load face features')
    
    args = parser.parse_args()

    print("Starting real-time face recognition system...")
    
    # Setup paths
    project_root = get_project_root()
    data_dir = get_data_dir()
    reference_dir = data_dir / "lfw_dataset/target"  # Removed reference-dir argument
    embeddings_file = args.embeddings_file or (data_dir / "embeddings.npy")
    
    # Load or create features
    try:
        reference_features = np.load(embeddings_file, allow_pickle=True).item()
    except FileNotFoundError:
        print("No existing features found. Creating new reference features...")
        name, features = create_stable_features(reference_dir)
        save_features(name, features, embeddings_file.parent)
        reference_features = np.load(embeddings_file, allow_pickle=True).item()

    # Try to open the camera
    cap = None
    for device_id in range(2):  # Try first two video devices
        cap = try_camera(device_id)
        if cap is not None:
            break
    
    if cap is None:
        print("Error: Could not open any camera")
        return

    # Get actual camera properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera properties: {actual_width}x{actual_height} @ {actual_fps}fps")

    print("Starting video capture. Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read frame")
            break
            
        frame_count += 1
        if frame_count % 30 == 0:  # Print FPS every 30 frames
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        results = process_image(frame, reference_features)
        
        # Draw results on frame with landmarks
        draw_results(frame, results)

        # Display the resulting frame
        cv2.imshow('Facial Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final statistics
    if frame_count > 0:
        total_time = time.time() - start_time
        average_fps = frame_count / total_time
        print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds")
        print(f"Average FPS: {average_fps:.2f}")

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()