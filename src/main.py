import cv2
import numpy as np
import os
import time
from reference_creation import ReferenceCreator
from human_detection import detect_faces
from feature_extraction import extract_features
from comparison import compare_embeddings
from thresholding import is_match

def try_camera(device_id):
    print(f"Attempting to open camera {device_id}...")
    
    # Check if device exists
    device_path = f"/dev/video{device_id}"
    if not os.path.exists(device_path):
        print(f"Camera device {device_path} does not exist")
        return None
        
    # Try to open the camera
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Could not open camera {device_id}")
        return None
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Try to read a test frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Could not read frame from camera {device_id}")
        cap.release()
        return None
        
    print(f"Successfully opened camera {device_id}")
    print(f"Frame shape: {frame.shape}")
    return cap

def main():
    print("Starting facial recognition system...")
    
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

    # Initialize ReferenceCreator and load embeddings
    reference_creator = ReferenceCreator('reference_images', 'embeddings.npy')
    try:
        reference_embeddings = np.load('embeddings.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print("No existing embeddings found. Creating new reference embeddings...")
        reference_creator.create_references()
        reference_embeddings = np.load('embeddings.npy', allow_pickle=True).item()

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

        # Convert frame from BGR to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Detect faces in the frame
            boxes = detect_faces(frame_rgb)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = frame[y1:y2, x1:x2]

                    # Extract features from the detected face
                    features = extract_features(face)
                    if features is not None:
                        # Compare with reference embeddings
                        for name, ref_embedding in reference_embeddings.items():
                            similarity = compare_embeddings(features, ref_embedding)
                            
                            # Check if the similarity exceeds the threshold
                            if is_match(similarity):
                                label = f"Match: {name}"
                                break
                        else:
                            label = "Unknown"

                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

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