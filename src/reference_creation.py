from deepface import DeepFace
import os
import cv2
import numpy as np

class ReferenceCreator:
    def __init__(self, reference_images_folder, embeddings_file):
        self.reference_images_folder = reference_images_folder
        self.embeddings_file = embeddings_file
        self.embeddings = {}

    def load_images(self):
        images = []
        for filename in os.listdir(self.reference_images_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.reference_images_folder, filename)
                img = cv2.imread(img_path)
                images.append((filename, img))
        return images

    def extract_embeddings(self, images):
        for filename, img in images:
            try:
                result = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
                if result:
                    self.embeddings[filename] = result[0]['embedding']
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    def save_embeddings(self):
        np.save(self.embeddings_file, self.embeddings)

    def create_references(self):
        images = self.load_images()
        self.extract_embeddings(images)
        self.save_embeddings()

if __name__ == "__main__":
    reference_creator = ReferenceCreator('path/to/reference/images', 'embeddings.npy')
    reference_creator.create_references()