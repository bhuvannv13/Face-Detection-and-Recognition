import cv2

def test_camera(index=0):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        print(f"Error: Could not open camera with index {index}")
        return False
    ret, frame = cam.read()
    if ret:
        cv2.imshow('Test Frame', frame)
        cv2.waitKey(0)
        cam.release()
        cv2.destroyAllWindows()
        return True
    else:
        print("Error: Failed to capture image.")
        cam.release()
        return False

# Test camera with default backend and index 0
if test_camera(0):
    print("Camera test successful.")
else:
    print("Camera test failed. Trying different backends...")

# Try different backends for VideoCapture
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
success = False

for backend in backends:
    cam = cv2.VideoCapture(0, backend)
    if cam.isOpened():
        print(f"Using backend {backend}")
        success = True
        break
    cam.release()

if not success:
    print("Error: Could not open any camera backend.")
    exit()

cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id: ')

print("\n [INFO] Initializing face capture....")
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break
    elif count >= 30:  # Take 30 face samples and stop video
        break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
