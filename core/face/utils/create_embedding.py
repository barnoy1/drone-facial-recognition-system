import os
import cv2
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.face.utils.face_utils import extract_features, get_person_features

def create_stable_features(positive_samples_dir):
    """
    Create stable features (embedding and landmarks) by averaging features from multiple images.
    
    Args:
        positive_samples_dir (str): Path to directory containing positive samples of target
        
    Returns:
        tuple: (name, features) where name is extracted from filename and features contain embedding and landmarks
    """
    print(f"Creating stable features from samples in {positive_samples_dir}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(positive_samples_dir).glob(f'*{ext}')))
    
    if not image_files:
        raise ValueError(f"No image files found in {positive_samples_dir}")
    
    # Extract features from each image
    all_embeddings = []
    all_landmarks = {}
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error reading image {img_path}")
            continue
            
        features = get_person_features(image)
        if features is not None:
            all_embeddings.append(features['encoding'])
            
            # Initialize landmark averages on first valid detection
            if not all_landmarks:
                all_landmarks = {
                    feature: {
                        'points': [[] for _ in points],
                        'count': 0
                    }
                    for feature, points in features['landmarks'].items()
                }
            
            # Accumulate landmark points
            for feature, points in features['landmarks'].items():
                for i, point in enumerate(points):
                    all_landmarks[feature]['points'][i].append(point)
                all_landmarks[feature]['count'] += 1
        else:
            print(f"No face detected in {img_path}")
    
    if not all_embeddings:
        raise ValueError("No valid features could be extracted from any image")
    
    # Create stable embedding by averaging
    stable_embedding = np.mean(all_embeddings, axis=0)
    stable_embedding = stable_embedding / np.linalg.norm(stable_embedding)
    
    # Create stable landmarks by averaging
    stable_landmarks = {
        feature: [
            (
                sum(p[0] for p in info['points'][i])/len(info['points'][i]),
                sum(p[1] for p in info['points'][i])/len(info['points'][i])
            ) 
            for i in range(len(info['points']))
        ]
        for feature, info in all_landmarks.items()
    }
    
    # Extract name from first image (assumes all images are of same person)
    name = image_files[0].stem.split('_')[0]  # Assumes format "Name_number.jpg"
    
    print(f"Created stable features for {name} from {len(all_embeddings)} images")
    return name, {
        'encoding': stable_embedding,
        'landmarks': stable_landmarks
    }

def save_features(name, features, output_dir='data'):
    """
    Save features (embedding and landmarks) to a numpy file.
    
    Args:
        name (str): Name of the target person
        features (dict): Feature dictionary containing embedding and landmarks
        output_dir (str): Directory to save features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create features dictionary
    reference_features = {name: features}
    
    # Save to file
    output_path = output_dir / 'embeddings.npy'
    np.save(output_path, reference_features)
    print(f"Saved features to {output_path}")

def main():
    """Command line interface for creating embeddings."""
    import argparse
    parser = argparse.ArgumentParser(description='Create stable features from positive samples')
    parser.add_argument('--input', required=True,
                      help='Directory containing positive samples')
    parser.add_argument('--output-dir', default='data',
                      help='Directory to save features (default: data)')
    
    args = parser.parse_args()
    
    # Create and save features
    name, features = create_stable_features(args.input)
    save_features(name, features, args.output_dir)

if __name__ == '__main__':
    main()
