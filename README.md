# tflite_object_detection
clone this repository

# Install Dependencies:
!pip install os-sys

!pip install cv2

!pip install numpy

!pip install psutils

!pip install importlib

!pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# RUN INFERENCE:

from inside the directory run object_detection.py

P.S: by deefault a video sample will be ran, inorder to change the video feed just change:

cap = cv2.VideoCapture('video_sample1.mp4')  --------to:

cap = cv2.VideoCapture(<video file path>) --------or to get the feed from your webcam,
 
cap = cv2.VideoCapture(0)
