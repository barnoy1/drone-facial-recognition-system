import cv2

def test_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Webcam opened successfully")
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame from webcam")
    else:
        print("Could not read frame from webcam")
    
    cap.release()

if __name__ == '__main__':
    test_webcam()
