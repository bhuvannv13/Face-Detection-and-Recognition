# Face-Detection-and-Recognition

UDEMY: https://www.udemy.com/share/103qXC3@LXZSWJ9l6Cp65PG61pnM015LCuOHaE4Y51xftePZDVsaSgGLiZlsSbMk6L49gu3Mvw==/

This project implements face detection and recognition using Python and OpenCV.

## Features

- **Face Detection**: Identifies human faces in images or video streams using Haar cascades.
- **Face Recognition**: Recognizes and labels detected faces based on trained data.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bhuvannv13/Face-Detection-and-Recognition.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd Face-Detection-and-Recognition
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: Ensure that OpenCV is installed. If not, install it using:

   ```bash
   pip install opencv-python
   ```

## Usage

1. **Prepare Training Data**:

   - Create a directory named `training-images`.
   - Within `training-images`, create subdirectories for each person, named after the individual.
   - Add images of the person to the respective subdirectory.

   For example:

   ```
   training-images/
   ├── Person_A/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── Person_B/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

2. **Train the Model**:

   Run the training script to process the images and train the recognition model:

   ```bash
   python training.py
   ```

   This will generate the necessary model files for recognition.

3. **Run Face Detection and Recognition**:

   To start the face detection and recognition on a video stream (e.g., webcam), execute:

   ```bash
   python camtestrunscipt.py
   ```

   The script will access the webcam, detect faces, and recognize them based on the trained model.

## Acknowledgments

- This project utilizes OpenCV's Haar cascades for face detection.
- Inspired by various open-source face recognition projects and tutorials.

## License

This project is licensed under the MIT License.

For more details, visit the [GitHub repository](https://github.com/bhuvannv13/Face-Detection-and-Recognition). 
