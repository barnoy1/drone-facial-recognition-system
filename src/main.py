import cv2
import numpy as np
from reference_creation import create_reference_embeddings
from human_detection import detect_faces
from feature_extraction import extract_features
from comparison import compare_embeddings
from thresholding import is_match

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Load reference embeddings
    reference_embeddings = create_reference_embeddings()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        boxes, _ = detect_faces(frame)

        for box in boxes:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            # Extract features from the detected face
            features = extract_features(face)

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

        # Display the resulting frame
        cv2.imshow('Facial Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()