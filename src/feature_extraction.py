from deepface import DeepFace
import cv2

def extract_features(cropped_face):
    """
    Extracts facial features from a cropped face image using the DeepFace library.

    Parameters:
    cropped_face (numpy.ndarray): The cropped face image.

    Returns:
    numpy.ndarray: A 128D feature vector representing the facial features.
    """
    features = DeepFace.represent(cropped_face, model_name='Facenet', enforce_detection=False)
    return features[0]['embedding'] if features else None