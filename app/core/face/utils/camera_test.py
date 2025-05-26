import cv2
import sys
import numpy as np

def test_camera():
    # Print debug info
    print("Python version:", sys.version)
    print("OpenCV version:", cv2.__version__)
    
    # Try to open camera with specific backend
    print("\nTrying to open camera with V4L2 backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("Failed to open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        
    if not cap.isOpened():
        print("Could not open any camera")
        return

    # Configure camera with known working parameters
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully!")
    print("Camera properties:")
    print(f"- FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"- Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"- Format: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
    
    try:
        # Create a window first
        cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
        
        print("\nStarting capture loop. Press 'q' to exit...")
        while True:
            # Try to read a frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            if frame is None:
                print("Captured frame is None")
                break
                
            # Display the frame
            cv2.imshow('Camera Test', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during capture: {str(e)}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed")

if __name__ == "__main__":
    test_camera()