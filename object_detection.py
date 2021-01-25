#necessary imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import numpy as np
import sys
from threading import Thread
import importlib.util
import tflite_runtime.interpreter as tflite
import psutil
import time
# import tensorflow as tf


min_config_threshhold = 0.45
resW = 720
resH = 480
imW, imH = int(resW), int(resH)

#list so that we can count the frames in total
def fps(inference_time):
    fps = 1/inference_time
    return fps


#load label
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

labels = load_labels("labelmap.txt")

interpreter = tflite.Interpreter(model_path="model4.tflite")
interpreter.allocate_tensors()
 #get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


#check the type of input tensor
floating_model = (input_details[0]['dtype'] == np.float32)

#instead of using parser as in the  original  script it is defined globally
input_mean = 127.5
input_std = 127.5
#initailizing video stream
cap = cv2.VideoCapture('video_sample1.mp4')
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(resW,resH))
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    #normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    inference_start = time.time()
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    # #applying non maximum suppresion to boxes
    # selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size = 5, iou_threshold=0.2,score_threshold=float(0.5), name=None)
    # boxes,classes,scores = non_max_suppression(selected_indices,boxes,classes,scores)
    inference_end = time.time()
     # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_config_threshhold) and (scores[i] <= 1.0)):

            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

                #measure relative distance and give coollsion warning
            mid_x = (boxes[i][3] + boxes[i][1]) / 2
            mid_y = (boxes[i][2] + boxes[i][0]) / 2
            apx_distance = round((1 - (boxes[i][3] - boxes[i][1]))**4,1)
            cv2.putText(frame,'{}'.format(apx_distance),(int(mid_x * resW),int(mid_y * resH)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)


            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # All the results have been drawn on the frame, so it's time to display it.
    # inference_end = time.time()
    inference_time = inference_end - inference_start
    frameps = fps(inference_time)
    cv2.putText(frame, str(frameps), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Object detector', frame)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

#observing memory usage
process = psutil.Process(os.getpid())
print('memory usage is',process.memory_info().rss)

cv2.destroyAllWindows()
