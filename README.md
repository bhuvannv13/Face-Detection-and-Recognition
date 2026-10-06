# Face Detection and Recognition

Real-time face detection and recognition from a webcam using Python and OpenCV. Faces are detected with a Haar cascade and recognised with an LBPH (Local Binary Patterns Histograms) model trained on your own captured images.

Built as a learning project while following an OpenCV course.

## How it works

| Step | Script | What it does |
|---|---|---|
| 1. Test camera | `camtestrunscipt.py` | Checks that the webcam can be opened. |
| 2. Capture faces | `Face_DF.py` | Asks for a numeric user id, then saves 30 grayscale face crops to `dataset/`. |
| 3. Train | `training.py` | Trains the LBPH recogniser on `dataset/` and saves it to `trainer/trainer.yml`. |
| 4. Recognise | `Face_Recg.py` | Runs live recognition and labels each face with a name and confidence. |

## Requirements

- Python 3
- A webcam

```bash
pip install -r requirements.txt
```

This installs `opencv-contrib-python` (not plain `opencv-python`) because the LBPH recogniser lives in the `cv2.face` module. It is pinned below version 5, which the scripts were written for.

## Usage

```bash
git clone https://github.com/bhuvannv13/Face-Detection-and-Recognition.git
cd Face-Detection-and-Recognition

python camtestrunscipt.py   # optional camera check
python Face_DF.py           # capture faces; repeat with a new id for each person
python training.py          # train the recogniser
python Face_Recg.py         # live recognition, press ESC to quit
```

To show names instead of ids, edit the `names` list in `Face_Recg.py` so that the position of each name matches the user id you entered during capture.

## Limitations

- Haar cascades work best on frontal, well-lit faces.
- LBPH is a classical method and is far less accurate than modern deep learning face recognition.
- Captured images and the trained model contain personal biometric data, so keep `dataset/` and `trainer/` out of version control.

## Acknowledgements

Uses OpenCV's `haarcascade_frontalface_default.xml`.
