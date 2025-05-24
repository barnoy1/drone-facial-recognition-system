import cv2
import numpy as np
import os
import time
import argparse
import logging
import warnings
from pathlib import Path
from utils.create_embedding import create_stable_embedding, save_embedding
from utils.face_utils import process_image, extract_features, get_target_name_from_dir

# Configure logging with colors
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(message)s',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Remove default handler to avoid duplicate logs
logger.propagate = False
if logger.handlers:
    logger.handlers = [handler]

# Configure warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def get_data_dir():
    """Get the data directory."""
    return get_project_root() / 'data'

def save_processed_image(image, results, stats_text, output_path, filename):
    """Save an image with detection overlays and statistics."""
    display_image = image.copy()
    
    for result in results:
        x1, y1, x2, y2 = result['box']
        is_match = result['name'] != 'Unknown' and result['similarity'] >= 0.65
        color = (0, 255, 0) if is_match else (0, 0, 255)
        
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
        label = f"{result['name']} ({result['similarity']:.2f})"
        cv2.putText(display_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.putText(display_image, stats_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path / filename), display_image)

def save_statistics_to_csv(stats, csv_path):
    """Save detection statistics to a CSV file."""
    import csv
    
    csv_stats = {k: v for k, v in stats.items() if k != 'confidence_scores'}
    csv_stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_stats)

def create_summary_visualization(stats, output_path):
    """Create and save a summary visualization of detection statistics."""
    summary_height = 300
    summary_width = 600
    summary = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
    
    metrics = [
        ("PD", stats['pd']),
        ("1-FAR", 1 - stats['far']),
        ("Precision", stats['precision']),
        ("F1", stats['f1_score'])
    ]
    
    bar_width = 100
    spacing = 40
    start_x = 50
    
    for i, (label, value) in enumerate(metrics):
        x = start_x + i * (bar_width + spacing)
        height = int(value * 200)
        
        cv2.rectangle(summary, 
                     (x, summary_height - 80 - height),
                     (x + bar_width, summary_height - 80),
                     (0, 255, 0),
                     -1)
        
        cv2.putText(summary,
                   f"{label}: {value:.2%}",
                   (x, summary_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (255, 255, 255),
                   1)
    
    cv2.putText(summary,
               "Detection Performance Summary",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)
    
    cv2.imwrite(str(output_path), summary)

def evaluate_detection_statistics(input_dir, reference_embeddings, output_dir):
    """Evaluate detection statistics and save processed images."""
    stats = {
        'total_samples': 0,
        'total_positives': 0,
        'total_negatives': 0,
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
        'pd': 0.0,       # Probability of Detection (TP/(TP+FN))
        'far': 0.0,      # False Acceptance Rate (FP/(FP+TN))
        'tnr': 0.0,      # True Negative Rate (TN/(TN+FP))
        'precision': 0.0, # Precision (TP/(TP+FP))
        'recall': 0.0,   # Recall (TP/(TP+FN))
        'f1_score': 0.0, # F1 Score (2*(precision*recall)/(precision+recall))
        'multiple_faces_detected': 0,
        'avg_confidence': 0.0,
        'confidence_scores_positive': [],  # Confidence scores for positive samples
        'confidence_scores_negative': [],  # Confidence scores for negative samples
        'confidence_scores': []           # All confidence scores
    }
    
    output_dir = Path(output_dir)
    processed_output_dir = output_dir / 'processed_samples'
    
    print("\nProcessing samples...")
    input_path = Path(input_dir)
    
    # Get both positive and negative samples
    positive_dir = input_path / 'positive_samples'
    negative_dir = input_path / 'negative_samples'
    
    if not positive_dir.exists() or not negative_dir.exists():
        raise ValueError(f"Dataset directory must contain 'positive_samples' and 'negative_samples' subdirectories")
    
    # Get all images from both directories
    positive_files = list(positive_dir.glob('*.jpg'))
    negative_files = list(negative_dir.glob('*.jpg'))
    image_files = positive_files + negative_files
    stats['total_samples'] = len(image_files)
    stats['total_positives'] = len(positive_files)
    stats['total_negatives'] = len(negative_files)
    
    print(f"Found {len(positive_files)} positive samples and {len(negative_files)} negative samples")
    
    for img_path in image_files:
        # Check if this is a positive sample (same person) based on filename
        is_positive_sample = any(ref_name in img_path.stem for ref_name in reference_embeddings.keys())
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        results = process_image(image, reference_embeddings)
        
        if len(results) > 1:
            stats['multiple_faces_detected'] += 1
        
        # Get highest confidence score for this image
        max_confidence = max([r['similarity'] for r in results], default=0)
        stats['confidence_scores'].append(max_confidence)
            
        # Consider it a match if any detected face has high similarity
        is_match = any(r['name'] != 'Unknown' and r['similarity'] >= 0.65 for r in results)
        
        # Store confidence scores separately for positive and negative samples
        if is_positive_sample:
            stats['confidence_scores_positive'].append(max_confidence)
        else:
            stats['confidence_scores_negative'].append(max_confidence)
        
        # For positive samples (includes target name), we want matches
        # For negative samples (different people), we don't want matches
        if is_positive_sample:
            if is_match:
                stats['true_positives'] += 1  # Correctly matched a positive sample
            else:
                stats['false_negatives'] += 1  # Failed to match a positive sample
        else:
            if is_match:
                stats['false_positives'] += 1  # Incorrectly matched a negative sample
            else:
                stats['true_negatives'] += 1  # Correctly rejected a negative sample
            
        # Save processed image with detection results
        stats_text = f"Match: {is_match}, Confidence: {max([r['similarity'] for r in results], default=0):.2f}"
        save_processed_image(image, results, stats_text, processed_output_dir, img_path.name)
    
    # Calculate final statistics
    total_positives = stats['true_positives'] + stats['false_negatives']
    total_negatives = stats['true_negatives'] + stats['false_positives']
    
    # Calculate Probability of Detection (PD/Recall) = TP/(TP + FN)
    if total_positives > 0:
        stats['pd'] = stats['true_positives'] / total_positives
        stats['recall'] = stats['pd']
    
    # Calculate False Acceptance Rate (FAR) = FP/(FP + TN)
    if total_negatives > 0:
        stats['far'] = stats['false_positives'] / total_negatives
    
    # Calculate Precision = TP/(TP + FP)
    if (stats['true_positives'] + stats['false_positives']) > 0:
        stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
    
    # Calculate F1 Score = 2 * (precision * recall)/(precision + recall)
    if stats['precision'] > 0 and stats['recall'] > 0:
        stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
    
    # Calculate True Negative Rate (TNR)
    if total_negatives > 0:
        stats['tnr'] = stats['true_negatives'] / total_negatives
    
    # Calculate average confidence scores
    if stats['confidence_scores']:
        stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
    if stats['confidence_scores_positive']:
        stats['avg_confidence_positive'] = sum(stats['confidence_scores_positive']) / len(stats['confidence_scores_positive'])
    if stats['confidence_scores_negative']:
        stats['avg_confidence_negative'] = sum(stats['confidence_scores_negative']) / len(stats['confidence_scores_negative'])
    
    # Save statistics
    save_statistics_to_csv(stats, output_dir / 'statistics.csv')
    create_summary_visualization(stats, output_dir / 'summary.png')
    
    return stats

def get_lfw_pairs(root_dir):
    """Get matched and mismatched pairs from LFW dataset."""
    root_dir = Path(root_dir)
    pairs = {
        'matched': [],
        'mismatched': []
    }
    
    # Read matched pairs
    matched_pairs = root_dir / 'lfw' / 'matchpairsDevTest.csv'
    mismatched_pairs = root_dir / 'lfw' / 'mismatchpairsDevTest.csv'
    
    if matched_pairs.exists():
        with open(matched_pairs, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
                name, img1, img2 = line.strip().split(',')
                pairs['matched'].append({
                    'name': name,
                    'img1': root_dir / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled' / name / img1,
                    'img2': root_dir / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled' / name / img2
                })
    
    if mismatched_pairs.exists():
        with open(mismatched_pairs, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
                name1, img1, name2, img2 = line.strip().split(',')
                pairs['mismatched'].append({
                    'name1': name1,
                    'name2': name2,
                    'img1': root_dir / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled' / name1 / img1,
                    'img2': root_dir / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled' / name2 / img2
                })
    
    return pairs

def setup_evaluation_dirs(input_dir, target_name=None):
    """Setup evaluation directories for positive and negative samples.
    
    Args:
        input_dir: Base input directory (can be LFW dataset or custom directory)
        target_name: Optional target person name for LFW dataset
        
    Returns:
        tuple: (positive_dir, negative_dir)
    """
    input_dir = Path(input_dir)
    
    # Check if this is LFW dataset
    if (input_dir / 'lfw').exists():
        print("Detected LFW dataset structure")
        if target_name:
            # Use specified person's directory as positive samples
            positive_dir = input_dir / 'lfw' / 'lfw-deepfunneled' / 'lfw-deepfunneled' / target_name
            # Use other random persons as negative samples
            negative_dir = input_dir / 'negative_samples'
            if not negative_dir.exists():
                negative_dir.mkdir(parents=True)
                # TODO: Randomly select images from other persons for negative samples
        else:
            # Use matched/mismatched pairs from LFW
            pairs = get_lfw_pairs(input_dir)
            return pairs
    else:
        # Standard directory structure with positive_samples and negative_samples
        positive_dir = input_dir if input_dir.name == 'positive_samples' else input_dir / 'positive_samples'
        negative_dir = input_dir.parent / 'negative_samples'
    
    if not positive_dir.exists():
        raise ValueError(f"Positive samples directory not found: {positive_dir}")
    if not negative_dir.exists():
        raise ValueError(f"Negative samples directory not found: {negative_dir}")
        
    return positive_dir, negative_dir

def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Face Recognition System')
    parser.add_argument('--input', type=str, required=True,
                      help='Directory containing test images')
    parser.add_argument('--reference-dir', type=str, required=True,
                      help='Directory containing reference images')
    parser.add_argument('--embeddings-file', type=str,
                      help='File to store/load face embeddings')
    parser.add_argument('--output-dir', type=str,
                      help='Directory to store evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for processing')
    
    args = parser.parse_args()
    
    logger.info("Starting evaluation mode...")
    
    # Setup paths
    project_root = get_project_root()
    data_dir = get_data_dir()
    embeddings_file = args.embeddings_file or (data_dir / 'embeddings.npy')
    output_dir = args.output_dir or (data_dir / f'evaluation_results/{time.strftime("%Y%m%d_%H%M%S")}')
    
    logger.info(f"Using embeddings file: {embeddings_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load or create embeddings
    try:
        reference_embeddings = np.load(embeddings_file, allow_pickle=True).item()
        logger.info(f"Loaded embeddings for: {list(reference_embeddings.keys())}")
    except FileNotFoundError:
        logger.info("No existing embeddings found. Creating new reference embeddings...")
        
    # Get input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Error: Input directory {input_dir} does not exist")
        return
    
    # List all image files
    image_files = list(input_dir.glob('*.jpg'))
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    stats = evaluate_detection_statistics(input_dir, reference_embeddings, output_dir)
    
    # Save summary statistics
    logger.info("\n=== Detection Statistics ===\n")
    logger.info("Sample Counts:")
    logger.info(f"Total samples processed: {stats['total_samples']}")
    logger.info(f"- Positive samples: {stats['total_positives']}")
    logger.info(f"- Negative samples: {stats['total_negatives']}")
    logger.info(f"Images with multiple faces: {stats['multiple_faces_detected']}\n")
    
    logger.info("Detection Results:")
    logger.info(f"True positives (TP): {stats['true_positives']}")
    logger.info(f"False positives (FP): {stats['false_positives']}")
    logger.info(f"True negatives (TN): {stats['true_negatives']}")
    logger.info(f"False negatives (FN): {stats['false_negatives']}\n")
    
    logger.info("Performance Metrics:")
    logger.info(f"Probability of Detection (PD/Recall): {stats['pd']:.2%}")
    logger.info(f"False Acceptance Rate (FAR): {stats['far']:.2%}")
    logger.info(f"True Negative Rate (TNR): {stats['tnr']:.2%}")
    logger.info(f"Precision: {stats['precision']:.2%}")
    logger.info(f"F1 Score: {stats['f1_score']:.2%}")
    logger.info("\nConfidence Scores:")
    logger.info(f"Overall Average: {stats['avg_confidence']:.2f}")
    if 'avg_confidence_positive' in stats:
        logger.info(f"Positive Samples Average: {stats['avg_confidence_positive']:.2f}")
    if 'avg_confidence_negative' in stats:
        logger.info(f"Negative Samples Average: {stats['avg_confidence_negative']:.2f}\n")
    
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()