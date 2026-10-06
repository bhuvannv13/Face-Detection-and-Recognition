"""Capture face samples from the webcam and save them to dataset/."""
import os
import sys

import cv2

DATASET_DIR = "dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
SAMPLES_PER_USER = 30


def open_camera(index=0):
    """Open the webcam, trying the default backend first and then fallbacks."""
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    for backend in backends:
        cam = cv2.VideoCapture(index, backend)
        if cam.isOpened():
            return cam
        cam.release()
    return None


def main():
    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        sys.exit(f"Error: could not load {CASCADE_PATH}")

    # For each person, enter one numeric face id
    face_id = input("\n enter user id: ").strip()
    if not face_id.isdigit():
        sys.exit("Error: user id must be a whole number, for example 1")

    cam = open_camera()
    if cam is None:
        sys.exit("Error: could not open the camera.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs(DATASET_DIR, exist_ok=True)
    print("\n [INFO] Initializing face capture. Look at the camera ...")
    count = 0

    try:
        while count < SAMPLES_PER_USER:
            ret, img = cam.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                # Save the captured face into the dataset folder
                path = os.path.join(DATASET_DIR, f"User.{face_id}.{count}.jpg")
                cv2.imwrite(path, gray[y:y + h, x:x + w])

            cv2.imshow("image", img)
            if cv2.waitKey(100) & 0xFF == 27:  # Press 'ESC' to exit
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print(f"\n [INFO] Saved {count} face samples for user {face_id}. Exiting Program")


if __name__ == "__main__":
    main()
