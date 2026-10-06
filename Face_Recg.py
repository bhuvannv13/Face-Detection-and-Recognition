"""Recognise faces from the webcam using the trained LBPH model."""
import os
import sys

import cv2

TRAINER_PATH = os.path.join("trainer", "trainer.yml")
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Position in this list = user id entered in Face_DF.py.
# Example: names = ["None", "Alice", "Bob"] labels id 1 as Alice and id 2 as Bob.
names = ["None", "User 1", "User 2", "User 3", "User 4", "User 5"]


def label_for(user_id):
    """Return the display name for a user id, falling back to the id itself."""
    if 0 <= user_id < len(names):
        return names[user_id]
    return f"id {user_id}"


def main():
    if not hasattr(cv2, "face"):
        sys.exit("Error: cv2.face is missing. Install it with: pip install opencv-contrib-python")
    if not os.path.exists(TRAINER_PATH):
        sys.exit(f"Error: {TRAINER_PATH} not found. Run training.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        sys.exit(f"Error: could not load {CASCADE_PATH}")

    font = cv2.FONT_HERSHEY_TRIPLEX

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit("Error: could not open the camera.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Define min window size to be recognized as a face
    min_w = int(0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    min_h = int(0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(min_w, min_h),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                user_id, distance = recognizer.predict(gray[y:y + h, x:x + w])

                # LBPH returns a distance: 0 is a perfect match, lower is better
                if distance < 100:
                    label = label_for(user_id)
                    confidence = "  {0}%".format(round(100 - distance))
                else:
                    label = "unknown"
                    confidence = ""

                cv2.putText(img, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow("camera", img)

            if cv2.waitKey(10) & 0xFF == 27:  # Press 'ESC' to exit
                break
    finally:
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program")
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
