import cv2
import mediapipe as mp
import os
import logging

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(-1)  # Use camera

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

try:
    os.makedirs("faces")  # Directory to save face images
except FileExistsError:
    pass

frame_count = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_image = image[y:y+h, x:x+w]
            cv2.imwrite(f"faces/face_{frame_count}.jpg", face_image)
            frame_count += 1
            logging.warning("saved %d images for training..",frame_count)


    if frame_count >= 50:  # Save 50 face images, you can change this number
        break

cap.release()
