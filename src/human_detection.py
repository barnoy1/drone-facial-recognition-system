from facenet_pytorch import MTCNN
import cv2

def detect_faces(frame):
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    boxes, _ = mtcnn.detect(frame)
    return boxes