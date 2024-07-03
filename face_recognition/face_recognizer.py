import cv2
import os
import numpy as np
import logging

model_path = "../trained_models/face_recognizer_model.yml"
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

class FaceRecognizer:
    
    def __init__(self, dataset_path, model_path):
        
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
        
        if os.path.exists(self.model_path):
            self.face_recognizer_model.read(self.model_path)
        
    def prepare_training_data(self):

        faces = []
        labels = []
        label_map = {}

        for label, person_name in enumerate(os.listdir(self.dataset_path)):
            person_path = os.path.join(self.dataset_path, person_name)
            label_map[label] = person_name

            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                labels.append(label)

        return faces, labels
    
    def train(self):
        faces, labels = self.prepare_training_data()
        
        self.face_recognizer_model.train(faces, np.array(labels))
        self.face_recognizer_model.write(self.model_path)
    
    def predict(self, face):

        label, confidence = self.face_recognizer_model.predict(face)
        
        return label, confidence
    

if __name__ == "__main__":
    
    face_recognizer = FaceRecognizer(dataset_path = "../dataset/", model_path = model_path)
    # face_recognizer.train()
    # logging.warning("train done!")

    image_path = 'test1.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    label, confidence = face_recognizer.predict(image)
    print(f"Predicted Label: {label}, Confidence: {confidence}")
