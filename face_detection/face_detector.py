import cv2
import argparse
import mediapipe as mp

cascade_folder_string = cv2.data.haarcascades


class FaceDetector:
    def __init__(self, cascade_path=cascade_folder_string + '/haarcascade_frontalface_default.xml'):
        # Load the Haar Cascade face detector (xml file)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        # Initialize MediaPipe Face Detection.
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.is_mp = False

    def detect_faces_haarcascade(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return faces, len(faces)

    def detect_faces_mp(self, image):
        self.is_mp = True

        # mediaPipe needs the image to be in the converted format
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)

        if results.detections:
            return results, len(results.detections)
        else:
            return None, None

    def draw_faces(self, image, faces):
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def draw_faces_mp(self, image, mp_processing_results):
        # Draw face detections of each face.
        if mp_processing_results.detections:
            for detection in mp_processing_results.detections:
                self.mp_drawing.draw_detection(image, detection)
   
    def display_result(self, image):
        cv2.imshow('Face Detection Media Pipe', image) if self.is_mp == True else cv2.imshow('Face Detection', image)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(description='Detect faces in an image.')
    parser.add_argument('image_path', help='Path to the input image.')

    args = parser.parse_args()

    face_detector = FaceDetector()

    image = cv2.imread(args.image_path)

    if image is None:
        print(f"Error: Unable to load image from {args.image_path}")
        return

    # Detect faces
    detected_faces, num_faces = face_detector.detect_faces_haarcascade(image)
    print(f"total {num_faces} faces found in the image")

    if num_faces == 0:
        # calling the media pipe for detecting faces
        mp_processing_results, num_faces = face_detector.detect_faces_mp(image)
        
        if num_faces is None:
            print("Error: Unable to detect any face with Media Pipe Library as well")
            return
        else:
            face_detector.draw_faces_mp(image = image, mp_processing_results = mp_processing_results)

    else:
        # Draw faces on the image
        face_detector.draw_faces(image, detected_faces)

    # Display the result
    face_detector.display_result(image)

if __name__ == "__main__":
    main()