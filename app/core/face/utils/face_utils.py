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

def get_person_features(image):
    """Extract both face features and landmarks using face_recognition library.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        dict: Dictionary containing face encoding and landmarks, or None if no face found
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image
        
    # Detect face locations first
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return None
        
    # Get face encodings and landmarks
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)[0]
    landmarks = face_recognition.face_landmarks(rgb_image, face_locations)[0]
    
    # Convert landmarks to list of tuples to prevent path-related issues
    processed_landmarks = {
        feature: [(float(x), float(y)) for (x, y) in points] 
        for feature, points in landmarks.items()
    }
    
    return {
        'encoding': face_encodings,
        'landmarks': processed_landmarks
    }

def extract_features(image):
    """Extract face features using face_recognition library.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        numpy.ndarray: Face encoding vector or None if no face found
    """
    features = get_person_features(image)
    if features is None:
        return None
    return features['encoding']

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

def process_image(image, reference_features):
    """Process a single image and return face detection and recognition results.
    
    Args:
        image: Input image in BGR format
        reference_features: Dictionary containing reference embeddings and landmarks
        
    Returns:
        list: List of dictionaries containing detection and recognition results
    """
    # Detect faces using cascade classifier first (faster)
    face_boxes = detect_faces(image)
    if face_boxes is None:
        return []
    
    results = []
    for box in face_boxes:
        # Extract face region
        x1, y1, x2, y2 = box
        face_image = image[y1:y2, x1:x2]
        
        # Get features
        features = get_person_features(face_image)
        if features is None:
            continue            # Compare with each reference
        best_match = {'name': 'Unknown', 'similarity': 0.0}
        for name, ref_features in reference_features.items():
            # Compare both embeddings and landmarks
            similarity = compare_features(features, ref_features)
            if similarity > best_match['similarity']:
                best_match = {'name': name, 'similarity': similarity}
        
        # Add detection and recognition results
        results.append({
            'box': box.tolist(),  # Convert numpy array to list
            'name': best_match['name'],
            'similarity': best_match['similarity'],
            'landmarks': features['landmarks'] if features else None
        })
    
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

def compare_landmarks(landmarks1, landmarks2):
    """Compare two sets of facial landmarks and return similarity score.
    
    Args:
        landmarks1: First set of facial landmarks
        landmarks2: Second set of facial landmarks
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if landmarks1 is None or landmarks2 is None:
        return 0.0
    
    total_distance = 0
    point_count = 0
    
    # Compare corresponding landmarks for each facial feature
    for feature in landmarks1.keys():
        if feature not in landmarks2:
            continue
            
        points1 = landmarks1[feature]
        points2 = landmarks2[feature]
        
        # Skip if different number of points
        if len(points1) != len(points2):
            continue
            
        # Calculate average distance between corresponding points
        for p1, p2 in zip(points1, points2):
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            total_distance += distance
            point_count += 1
    
    if point_count == 0:
        return 0.0
        
    # Convert average distance to similarity score (0 to 1)
    avg_distance = total_distance / point_count
    similarity = 1 / (1 + avg_distance/50)  # Normalize by typical face size (adjusted for better sensitivity)
    return float(similarity)

def compare_features(features1, features2, embedding_weight=0.85):
    """Compare two feature sets using both embeddings and landmarks.
    
    Args:
        features1: First feature set containing encoding and landmarks
        features2: Second feature set containing encoding and landmarks
        embedding_weight: Weight given to embedding similarity (0 to 1, default 0.85)
        
    Returns:
        float: Combined similarity score between 0 and 1
    """
    if features1 is None or features2 is None:
        return 0.0
        
    # Compare embeddings
    embedding_similarity = compare_embeddings(
        features1.get('encoding'),
        features2.get('encoding')
    )
    
    # Compare landmarks
    landmark_similarity = compare_landmarks(
        features1.get('landmarks'),
        features2.get('landmarks')
    )
    
    # Weighted average of similarities
    landmark_weight = 1 - embedding_weight
    combined_similarity = (embedding_similarity * embedding_weight + 
                         landmark_similarity * landmark_weight)
    
    return float(combined_similarity)
