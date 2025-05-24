import os
import cv2
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.face_utils import extract_features

def create_stable_embedding(positive_samples_dir):
    """
    Create a stable embedding by averaging features from multiple images of the target.
    
    Args:
        positive_samples_dir (str): Path to directory containing positive samples of target
        
    Returns:
        tuple: (name, embedding) where name is extracted from filename and embedding is numpy array
    """
    print(f"Creating stable embedding from samples in {positive_samples_dir}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(positive_samples_dir).glob(f'*{ext}')))
    
    if not image_files:
        raise ValueError(f"No image files found in {positive_samples_dir}")
    
    # Extract features from each image
    all_features = []
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error reading image {img_path}")
            continue
            
        features = extract_features(image)
        if features is not None:
            all_features.append(features)
        else:
            print(f"No face detected in {img_path}")
    
    if not all_features:
        raise ValueError("No valid features could be extracted from any image")
    
    # Create stable embedding by averaging all features
    stable_embedding = np.mean(all_features, axis=0)
    
    # Normalize the embedding
    stable_embedding = stable_embedding / np.linalg.norm(stable_embedding)
    
    # Extract name from first image (assumes all images are of same person)
    name = image_files[0].stem.split('_')[0]  # Assumes format "Name_number.jpg"
    
    print(f"Created stable embedding for {name} from {len(all_features)} images")
    return name, stable_embedding

def save_embedding(name, embedding, output_dir='data'):
    """
    Save embedding to a numpy file.
    
    Args:
        name (str): Name of the target person
        embedding (numpy.ndarray): Feature embedding
        output_dir (str): Directory to save embeddings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings dictionary
    embeddings = {name: embedding}
    
    # Save to file
    output_path = output_dir / 'embeddings.npy'
    np.save(output_path, embeddings)
    print(f"Saved embedding to {output_path}")

def main():
    """Command line interface for creating embeddings."""
    import argparse
    parser = argparse.ArgumentParser(description='Create stable embedding from positive samples')
    parser.add_argument('--input', required=True,
                      help='Directory containing positive samples')
    parser.add_argument('--output-dir', default='data',
                      help='Directory to save embeddings (default: data)')
    
    args = parser.parse_args()
    
    # Create and save embedding
    name, embedding = create_stable_embedding(args.input)
    save_embedding(name, embedding, args.output_dir)

if __name__ == '__main__':
    main()
