# Face Movement Tracker

## Overview

Face Movement Tracker is a Python-based project that monitors the duration a person spends in front of a camera by tracking their face movements. This can be used for various applications such as productivity tracking, screen time analysis, or security monitoring.

## Features

Real-time face detection using OpenCV and dlib/MediaPipe.

Tracks the duration a person remains in front of the camera.

Logs entry and exit timestamps.

Optional data visualization for monitoring trends.

## Technologies Used

Python

OpenCV (for face detection and tracking)

dlib/MediaPipe (for facial landmark detection)

NumPy (for data processing)

Matplotlib (for data visualization)

## Installation

### Clone the repository:

git clone https://github.com/belepod/FaceTime-Tracker
cd face-movement-tracker

### Install dependencies:

pip install -r requirements.txt

### Usage

Run the script:

python main.py
python maingraph.py

The script will start tracking and logging the duration for which a face is detected.

Press q to stop tracking and view the results.

## Configuration

You can adjust parameters like face detection sensitivity and logging frequency in main.py.

Future Enhancements

Add support for multiple face tracking.

Improve accuracy using deep learning models.

Web-based dashboard for visualization.

## License

This project is licensed under the MIT License.
