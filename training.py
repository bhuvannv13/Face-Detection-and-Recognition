"""Train the LBPH face recogniser on the images in dataset/."""
import os
import sys

import cv2
import numpy as np
from PIL import Image

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_images_and_labels(path, detector):
    """Return face crops and their numeric user ids from files named User.<id>.<n>.jpg."""
    face_samples = []
    ids = []

    for file_name in sorted(os.listdir(path)):
        if not file_name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        try:
            user_id = int(file_name.split(".")[1])
        except (IndexError, ValueError):
            print(f" [WARN] Skipping {file_name}: expected a name like User.1.5.jpg")
            continue

        pil_img = Image.open(os.path.join(path, file_name)).convert("L")  # grayscale
        img_numpy = np.array(pil_img, "uint8")

        for (x, y, w, h) in detector.detectMultiScale(img_numpy):
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(user_id)

    return face_samples, ids


def main():
    if not hasattr(cv2, "face"):
        sys.exit("Error: cv2.face is missing. Install it with: pip install opencv-contrib-python")
    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"Error: '{DATASET_DIR}' folder not found. Run Face_DF.py first.")

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        sys.exit(f"Error: could not load {CASCADE_PATH}")

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(DATASET_DIR, detector)
    if not faces:
        sys.exit("Error: no faces found in the dataset. Capture some with Face_DF.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    os.makedirs(TRAINER_DIR, exist_ok=True)
    recognizer.write(os.path.join(TRAINER_DIR, "trainer.yml"))

    print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))


if __name__ == "__main__":
    main()
