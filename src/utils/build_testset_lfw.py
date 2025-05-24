import os
import random
import shutil
import pandas as pd
import zipfile
import cv2
import numpy as np
from pathlib import Path
import face_recognition
from tqdm import tqdm

# DJI Tello camera resolution
TELLO_WIDTH = 960
TELLO_HEIGHT = 720

def process_image_for_tello(source_path, target_path):
    """Process an image to match DJI Tello camera specifications."""
    try:
        # Read the image
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"Error: Could not read image {source_path}")
            return False
            
        # Detect face in the image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print(f"Warning: No face detected in {source_path}")
            # If no face detected, just resize the whole image
            resized = cv2.resize(image, (TELLO_WIDTH, TELLO_HEIGHT))
        else:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Add padding around the face
            padding_x = int(w * 0.5)  # 50% padding on each side
            padding_y = int(h * 0.5)  # 50% padding on top and bottom
            
            # Calculate new coordinates with padding
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(image.shape[1], x + w + padding_x)
            y2 = min(image.shape[0], y + h + padding_y)
            
            # Crop the image
            face_img = image[y1:y2, x1:x2]
            
            # Resize to Tello resolution while maintaining aspect ratio
            aspect_ratio = TELLO_WIDTH / TELLO_HEIGHT
            current_ratio = face_img.shape[1] / face_img.shape[0]
            
            if current_ratio > aspect_ratio:
                # Image is wider than target ratio
                new_width = TELLO_WIDTH
                new_height = int(TELLO_WIDTH / current_ratio)
            else:
                # Image is taller than target ratio
                new_height = TELLO_HEIGHT
                new_width = int(TELLO_HEIGHT * current_ratio)
                
            resized = cv2.resize(face_img, (new_width, new_height))
            
            # Create black canvas of Tello size
            final_img = np.zeros((TELLO_HEIGHT, TELLO_WIDTH, 3), dtype=np.uint8)
            
            # Calculate position to paste resized image
            x_offset = (TELLO_WIDTH - new_width) // 2
            y_offset = (TELLO_HEIGHT - new_height) // 2
            
            # Paste resized image onto black canvas
            final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            resized = final_img
        
        # Save the processed image
        cv2.imwrite(str(target_path), resized)
        return True
        
    except Exception as e:
        print(f"Error processing image {source_path}: {str(e)}")
        return False

def get_person_features(image_path):
    """Extract features from a person's image."""
    try:
        # Convert path to string to ensure compatibility
        image_path_str = str(image_path)
        image = face_recognition.load_image_file(image_path_str)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"No face detected in {image_path_str}")
            return None
        
        # Get facial features
        try:
            face_encodings = face_recognition.face_encodings(image, face_locations)[0]
            landmarks = face_recognition.face_landmarks(image, face_locations)[0]
            
            # Convert landmarks to list of tuples to prevent path-related issues
            processed_landmarks = {
                feature: [(float(x), float(y)) for (x, y) in points] 
                for feature, points in landmarks.items()
            }
            
            return {
                'encoding': face_encodings,
                'landmarks': processed_landmarks
            }
        except IndexError:
            print(f"Failed to extract features from detected face in {image_path_str}")
            return None
    except Exception as e:
        print(f"Error processing image {image_path_str}: {str(e)}")
        return None

def estimate_similarity(target_features, candidate_features):
    """Estimate similarity between two people based on facial features."""
    if target_features is None or candidate_features is None:
        return 0.0
    
    # Face encoding similarity (already normalized between 0-1)
    face_distance = face_recognition.face_distance(
        [target_features['encoding']], 
        candidate_features['encoding']
    )[0]
    face_similarity = 1.0 - min(face_distance, 1.0)  # Ensure between 0-1
    
    # Convert landmark points to numpy arrays for proper calculation
    target_landmarks = np.array([point for feature in target_features['landmarks'].values() for point in feature])
    candidate_landmarks = np.array([point for feature in candidate_features['landmarks'].values() for point in feature])

    # Calculate distances between corresponding landmarks
    distances = np.linalg.norm(target_landmarks - candidate_landmarks, axis=1)
    
    # Normalize distances to 0-1 range using min-max normalization
    # Most facial landmarks are within the face bounds (0-1 in face_recognition coordinates)
    normalized_distances = np.clip(distances / np.sqrt(2), 0, 1)  # sqrt(2) is max possible distance in normalized coordinates
    landmark_similarity = 1.0 - np.mean(normalized_distances)
    
    # Combine similarities with weights (70% face encoding, 30% landmarks)
    # Both components are now guaranteed to be between 0 and 1
    similarity = 0.7 * face_similarity + 0.3 * landmark_similarity
    
    # Final safety clamp to ensure output is between 0 and 1
    return float(np.clip(similarity, 0.0, 1.0))

def cleanup_directories(directories):
    """Clean up the specified directories."""
    for dir_path in directories:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Cleaned up: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)

def extract_lfw_dataset(base_dir):
    """Extract LFW dataset if not already present."""
    lfw_zip = base_dir / 'lfw.zip'
    lfw_dir = base_dir / 'lfw'
    
    # If LFW directory doesn't exist or is empty
    if not lfw_dir.exists() or not any(lfw_dir.iterdir()):
        if not lfw_zip.exists():
            print("Error: lfw.zip not found in the dataset directory")
            return False
            
        print("Extracting LFW dataset...")
        try:
            with zipfile.ZipFile(lfw_zip, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            print("LFW dataset extracted successfully")
            return True
        except Exception as e:
            print(f"Error extracting LFW dataset: {str(e)}")
            return False
    return True

def find_similar_people(target_person_dir, other_people, lfw_dir, num_samples, max_age_diff=8, similarity_threshold=0.3):
    """Find similar people based on facial features and age."""
    similar_people = []
    
    # Get features of target person
    target_images = list(target_person_dir.glob('*.jpg'))
    if not target_images:
        print(f"No images found in {target_person_dir}")
        return []
    
    print("\nExtracting features from target person...")
    target_features = get_person_features(target_images[0])
    if target_features is None:
        print("Failed to extract features from target person")
        return []
    
    print("\nScanning dataset for similar faces...")
    # Process each candidate with progress bar
    pbar = tqdm(total=len(other_people), desc="Analyzing faces")
    for _, person in other_people.iterrows():
        try:
            if len(similar_people) >= num_samples:
                print(f"\nFound requested number of similar faces ({num_samples}). Stopping search.")
                break
            
            # Skip invalid person records
            if pd.isna(person['name']) or not isinstance(person['name'], str):
                print(f"\nSkipping invalid person record: {person}")
                pbar.update(1)
                continue
                
            person_dir = lfw_dir / person['name']
            if not person_dir.exists():
                pbar.update(1)
                continue
                
            person_images = list(person_dir.glob('*.jpg'))
            if not person_images:
                pbar.update(1)
                continue
                
            # Get candidate features
            candidate_features = get_person_features(person_images[0])
            if candidate_features is None:
                pbar.update(1)
                continue
                
            # Calculate similarity
            try:
                similarity_score = estimate_similarity(target_features, candidate_features)
                
                if similarity_score > similarity_threshold:
                    similar_people.append({
                        'person': person,
                        'similarity': float(similarity_score),  # Ensure score is a float
                        'images': person_images
                    })
                    print(f"\nSimilar person found: {person['name']} (similarity: {similarity_score:.2f})")
            except Exception as e:
                print(f"\nError calculating similarity for {person['name']}: {str(e)}")
                pbar.update(1)
                continue
            
            pbar.update(1)
        except Exception as e:
            print(f"\nError processing person record: {str(e)}")
            pbar.update(1)
            continue
    
    pbar.close()
    
    # Sort by similarity and take top N
    similar_people.sort(key=lambda x: x['similarity'], reverse=True)
    found_samples = len(similar_people)
    actual_samples = min(num_samples, found_samples)
    
    if found_samples < num_samples:
        print(f"\nNote: Found only {found_samples} similar faces (requested {num_samples})")
    else:
        print(f"\nFound {found_samples} similar faces")
        
    return [(x['person'], x['similarity']) for x in similar_people[:actual_samples]]

def setup_dataset(num_positive_samples=5, num_negative_samples=5, 
                  output_dir='lfw_dataset', max_age_diff=8, similarity_threshold=0.3,
                  target_person_name=None):
    """
    Set up a facial recognition dataset using local LFW dataset.
    
    Args:
        num_positive_samples (int): Number of positive samples required
        num_negative_samples (int): Number of negative samples required
        output_dir (str): Directory to store the organized dataset
        max_age_diff (int): Maximum age difference for negative samples
        similarity_threshold (float): Similarity threshold for negative samples
        target_person_name (str, optional): Name of target person. If None, a random person will be selected
    
    Returns:
        tuple: (success status, target person name if successful)
    """
    # Setup paths
    base_dir = Path(output_dir)
    lfw_dir = base_dir / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled'
    people_csv = base_dir / 'lfw' / 'people.csv'
    
    # Create and clean output directories
    target_dir = base_dir / "target"
    positive_dir = base_dir / "positive_samples"
    negative_dir = base_dir / "negative_samples"
    temp_dir = base_dir / "temp"
    
    # Clean up previous run
    cleanup_directories([target_dir, positive_dir, negative_dir, temp_dir])
    
    # Extract LFW dataset if needed
    if not extract_lfw_dataset(base_dir):
        return False, None

    # Read and process people.csv
    try:
        df = pd.read_csv(people_csv)
        # Filter people who have enough images
        eligible_people = df[df['images'] >= num_positive_samples + 1]  # +1 for target image
        
        if target_person_name is not None:
            # Find the specified person
            target_person_data = eligible_people[eligible_people['name'] == target_person_name]
            if target_person_data.empty:
                available_images = df[df['name'] == target_person_name]['images'].iloc[0] if target_person_name in df['name'].values else 0
                if available_images == 0:
                    print(f"Error: Person '{target_person_name}' not found in dataset")
                else:
                    print(f"Error: Person '{target_person_name}' has only {available_images} images, need at least {num_positive_samples + 1}")
                return False, None
            target_person = target_person_data.iloc[0]
        else:
            # Randomly select a target person
            if eligible_people.empty:
                print(f"No person found with {num_positive_samples + 1} or more images")
                return False, None
            target_person = eligible_people.sample(n=1).iloc[0]
            target_person_name = target_person['name']
        
        target_person_dir = lfw_dir / target_person_name
        
        print(f"\nSelected target person: {target_person_name}")
        print(f"Available images: {target_person['images']}")
        
        # Get all images for the target person
        target_images = list(target_person_dir.glob('*.jpg'))
        if len(target_images) < num_positive_samples + 1:
            print(f"Error: Not enough images found in directory for {target_person_name}")
            return False, None
            
        # Process target image
        target_image = random.choice(target_images)
        target_images.remove(target_image)
        target_path = target_dir / f"{target_person_name}.jpg"
        if process_image_for_tello(target_image, target_path):
            print(f"Processed target image: {target_image.name}")
        
        # Process positive samples
        selected_positives = random.sample(target_images, num_positive_samples)
        for i, img_path in enumerate(selected_positives, 1):
            output_path = positive_dir / f"{target_person_name}_{i}.jpg"
            if process_image_for_tello(img_path, output_path):
                print(f"Processed positive sample {i}: {img_path.name}")
        
        # Find similar people for negative samples
        print("\nFinding similar people for negative samples...")
        other_people = df[df['name'] != target_person_name]
        similar_people = find_similar_people(
            target_person_dir, 
            other_people, 
            lfw_dir, 
            num_negative_samples, 
            max_age_diff, 
            similarity_threshold=similarity_threshold
        )
        
        # Continue even if we found fewer similar people than requested
        num_found_negatives = len(similar_people)
        
        for i, (person, similarity_score) in enumerate(similar_people, 1):
            person_dir = lfw_dir / person['name']
            person_images = list(person_dir.glob('*.jpg'))
            if person_images:
                neg_image = random.choice(person_images)
                output_path = negative_dir / f"different_person_{i}.jpg"
                if process_image_for_tello(neg_image, output_path):
                    print(f"Processed negative sample {i} from: {person['name']} (similarity: {similarity_score:.2f})")
        
        print("\nDataset organization complete:")
        print(f"- Target image: {target_dir}")
        print(f"- Positive samples: {positive_dir} ({num_positive_samples} images)")
        print(f"- Negative samples: {negative_dir} ({num_found_negatives} images)")
        print(f"\nAll images processed to match Tello camera resolution: {TELLO_WIDTH}x{TELLO_HEIGHT}")
        
        return True, target_person_name
        
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        return False, None

def get_lfw_path():
    """Get the path to the LFW dataset."""
    base_path = Path(__file__).resolve().parent.parent.parent
    lfw_path = base_path / 'data' / 'lfw_dataset' / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled'
    if not lfw_path.exists():
        raise ValueError(f"LFW dataset not found at {lfw_path}")
    return lfw_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Set up face recognition dataset from local LFW dataset')
    parser.add_argument('--output-dir', default='lfw_dataset',
                      help='Output directory (should contain the lfw directory)')
    parser.add_argument('--num-positive', type=int, default=5,
                      help='Number of positive samples')
    parser.add_argument('--num-negative', type=int, default=5,
                      help='Number of negative samples')
    parser.add_argument('--max-age-diff', type=int, default=8,
                      help='Maximum age difference for negative samples')
    parser.add_argument('--similarity', type=float, default=0.3,
                      help='Similarity threshold for negative samples')
    parser.add_argument('--target-person', type=str, default=None,
                      help='Name of the target person (e.g., "Kim_Dae-jung"). If not provided, a random person will be selected')
    
    args = parser.parse_args()
    success, target_person = setup_dataset(
        num_positive_samples=args.num_positive,
        num_negative_samples=args.num_negative,
        output_dir=args.output_dir,
        max_age_diff=args.max_age_diff,
        similarity_threshold=args.similarity,
        target_person_name=args.target_person
    )
    
    if not success:
        print("\nFailed to set up dataset")
        exit(1)

if __name__ == "__main__":
    main()