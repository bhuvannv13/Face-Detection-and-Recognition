import cv2

def test_camera_access(index=0):
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

# Test with the default index
if test_camera_access(0):
    print("Camera test successful.")
else:
    print("Camera test failed.")
