import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
import cv2

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll'
# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'
pyK4A = pyKinectAzure(modulePath)
# Open device
pyK4A.device_open()
# Modify camera configuration
device_config = pyK4A.config
device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
print(device_config)

# Start cameras using modified configuration
pyK4A.device_start_cameras(device_config)
cap = cv2.VideoCapture(0)
while True:
   ret,frame = cap.read()
   cv2.imshow('',frame)

   cv2.waitKey(30)