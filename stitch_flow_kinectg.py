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
device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
print(device_config)

# Start cameras using modified configuration
pyK4A.device_start_cameras(device_config)

def img_add(img1, img2):
    # print(img1.shape,img2.shape)
    img_new = np.ndarray([img1.shape[0], img1.shape[1] + img2.shape[1], 3],dtype=np.uint8)
    img_new[:, 0:img1.shape[1]] = img1
    img_new[:, img1.shape[1]:] = img2
    return np.array(img_new)
cap = cv2.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv2.resize(old_frame,(512,512))



old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
left_frame = old_frame.copy()
point_per = []
distance = 0
distance_per = 0

# g3 = np.array(p0.shape)
while(1):


    ret,frame = cap.read()
    frame  = cv2.resize(frame,(512,512))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # print(np.mean(good_new[0]))


    point_per = []
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):

        a,b = new.ravel()
        c,d = old.ravel()

        point_per.append(abs(c-a))
            # print(c-a)
        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # print(np.array(point_per))
    distance_per = np.mean(point_per)
    # print(distance_per)
    distance = distance + abs(distance_per)
    # print(len(point_per))

    p0 = good_new.reshape(-1,1,2)

    if distance > 1:
        right_frame = frame[:,int(distance)*(-1):]
        left_frame = img_add(left_frame,right_frame)

        # left_frame = old_frame.copy()
        distance = 0
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    img = cv2.add(frame,mask)
    cv2.imshow('img',img)
    cv2.imshow('result',left_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('pinjie.jpg',left_frame)
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()


cv2.destroyAllWindows()
cap.release()