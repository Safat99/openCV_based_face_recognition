import cv2
import os
import logging

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(-1)  # Use camera

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


try:
    os.makedirs("faces_haarcascade")  # Directory to save face images
except FileExistsError:
    pass

frame_count = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        cv2.imwrite(f"faces/face_{frame_count}.jpg", face_image)
        frame_count += 1
        logging.warning("saved %d images for training..",frame_count)


    if frame_count >= 50:  # Save 50 face images, you can change this number
        break

cap.release()
