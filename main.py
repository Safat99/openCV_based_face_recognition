from face_detection.face_detector import FaceDetector
import argparse
import cv2
import mediapipe as mp

cascade_folder_string = cv2.data.haarcascades


faceDetector = FaceDetector()
parser = argparse.ArgumentParser(description='Detect faces in an image.')
parser.add_argument('image_path', help='Path to the input image.')

args = parser.parse_args()

face_detector = FaceDetector()

image = cv2.imread(args.image_path)

if image is None:
    print(f"Error: Unable to load image from {args.image_path}")

# Detect faces
detected_faces, num_faces = face_detector.detect_faces_haarcascade(image)
print(f"total {num_faces} faces found in the image")

if num_faces == 0:
    # calling the media pipe for detecting faces
    mp_processing_results, num_faces = face_detector.detect_faces_mp(image)
    
    if num_faces is None:
        print("Error: Unable to detect any face with Media Pipe Library as well")
    else:
        face_detector.draw_faces_mp(image = image, mp_processing_results = mp_processing_results)

else:
    # Draw faces on the image
    face_detector.draw_faces(image, detected_faces)

# Display the result
face_detector.display_result(image)
