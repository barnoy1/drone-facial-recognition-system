"""Utilities for visualizing face detection and recognition results."""

import cv2
import numpy as np

def draw_landmarks(frame, landmarks, box, color=(0, 255, 0)):
    """Draw facial landmarks on the frame.
    
    Args:
        frame: Input frame to draw on
        landmarks: Dictionary of facial landmarks
        box: Bounding box coordinates [x1, y1, x2, y2]
        color: Color for drawing landmarks (B,G,R)
    """
    if landmarks is None:
        return
        
    # Scale landmark points based on face box
    x1, y1, x2, y2 = box
    face_width = x2 - x1
    face_height = y2 - y1
    
    for feature, points in landmarks.items():
        # Draw points for each facial feature
        for point in points:
            # Scale point coordinates to frame size
            x = int(x1 + point[0] * face_width)
            y = int(y1 + point[1] * face_height)
            cv2.circle(frame, (x, y), 1, color, -1)
            
        # Draw lines connecting points
        if len(points) > 1:
            pts = np.array([(int(x1 + p[0] * face_width), 
                           int(y1 + p[1] * face_height)) for p in points])
            cv2.polylines(frame, [pts], False, color, 1)

def draw_results(frame, results):
    """Draw detection and recognition results on frame.
    
    Args:
        frame: Input frame to draw on
        results: List of detection and recognition results
    """
    for result in results:
        # Draw bounding box
        x1, y1, x2, y2 = result['box']
        color = (0, 255, 0) if result['similarity'] >= 0.60 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw name and confidence
        label = f"{result['name']} ({result['similarity']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                   
        # Draw facial landmarks
        if result['landmarks']:
            draw_landmarks(frame, result['landmarks'], result['box'], color)
