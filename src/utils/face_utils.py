import numpy as np
import face_recognition
import cv2
import os

# Initialize the cascade classifier
cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame):
    """
    Detect faces using OpenCV's cascade classifier.
    Much faster than deep learning-based detectors on CPU.
    
    Args:
        frame: Input image in BGR format
        
    Returns:
        numpy.ndarray: Array of face bounding boxes in format [[x1,y1,x2,y2],...]
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
        
    # Convert format from [x,y,w,h] to [x1,y1,x2,y2]
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append([x, y, x+w, y+h])
        
    return np.array(boxes)

def extract_features(image):
    """Extract face features using face_recognition library.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        numpy.ndarray: Face encoding vector or None if no face found
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image
        
    # Get face encodings
    encodings = face_recognition.face_encodings(rgb_image)
    if not encodings:
        return None
    return encodings[0]

def compare_embeddings(embedding1, embedding2):
    """Compare two face embeddings and return similarity score.
    
    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
        
    # Calculate Euclidean distance and convert to similarity score
    distance = np.linalg.norm(embedding1 - embedding2)
    similarity = 1 / (1 + distance)
    return float(similarity)

def is_match(similarity_score, threshold=0.6):
    """Determine if similarity score indicates a match.
    
    Args:
        similarity_score: Float between 0 and 1
        threshold: Minimum similarity for a match
        
    Returns:
        bool: True if similarity exceeds threshold
    """
    return similarity_score >= threshold

def process_image(image, reference_embeddings):
    """Process a single image and return face detection and recognition results.
    
    Args:
        image: Input image in BGR format
        reference_embeddings: Dictionary of reference face embeddings
        
    Returns:
        list: List of dictionaries containing detection results
    """
    results = []
    try:
        # Convert frame from BGR to RGB for face detection
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Detect all faces in the frame
        boxes = detect_faces(image_rgb)
        if boxes is not None:
            for box in boxes:
                # Ensure all box coordinates are integers
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                # Skip if face region is too small
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                
                face = image[y1:y2, x1:x2]
                
                # Extract features from the detected face
                features = extract_features(face)
                if features is not None:
                    # Find best match among reference embeddings
                    best_match = {
                        'name': 'Unknown',
                        'similarity': 0.0
                    }
                    
                    # Compare with all reference embeddings
                    for name, ref_embedding in reference_embeddings.items():
                        similarity = compare_embeddings(features, ref_embedding)
                        
                        if similarity > best_match['similarity']:
                            best_match = {
                                'name': name,
                                'similarity': similarity
                            }
                    
                    # Add result with match status
                    results.append({
                        'box': (x1, y1, x2, y2),
                        'name': best_match['name'],
                        'similarity': float(best_match['similarity']),
                        'is_match': is_match(best_match['similarity'])
                    })

    except Exception as e:
        print(f"Error processing image: {e}")
    
    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

def get_target_name_from_dir(target_dir):
    """Extract target person's name from target directory.
    
    Args:
        target_dir: Path to directory containing target image
        
    Returns:
        str: Name of target person from image filename
    """
    target_files = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not target_files:
        raise ValueError(f"No image files found in {target_dir}")
    # Get first image file and remove extension
    target_name = os.path.splitext(target_files[0])[0]
    return target_name
