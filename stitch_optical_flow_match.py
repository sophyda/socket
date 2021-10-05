import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


ori_arr = []
rans_arr = []

def img_add(img1, img2):
    # print(img1.shape,img2.shape)
    img_new = np.ndarray([img1.shape[0], img1.shape[1] + img2.shape[1], 3],dtype=np.uint8)
    img_new[:, 0:img1.shape[1]] = img1
    img_new[:, img1.shape[1]:] = img2
    return np.array(img_new)

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



cap = cv2.VideoCapture(1)

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
left_frame = old_frame.copy()
# point_per = []
distance = 0
distance_per = 0

distance_row_per = 0
distance_row = 0
# g3 = np.array(p0.shape)
while(1):

    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # print(np.mean(good_new[0]))


    point_per = []
    point_row_per = []
    pts_old = []
    pts_new = []
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):

        a,b = new.ravel()
        c,d = old.ravel()
        pts_old.append([c,d])   # a
        pts_new.append([a,b])   # b

        point_per.append(abs(c-a))

    distance_per = np.mean(point_per)
    distance = distance + abs(distance_per)






    # print(len(point_per))

    p0 = good_new.reshape(-1,1,2)
    if distance >1:

        pts_new = np.array(pts_new)
        pts_old = np.array(pts_old)
        (H, status) = cv2.findHomography(pts_old, pts_new, cv2.RANSAC,
                                         4.0)

        print(H)



        right_frame = cv2.warpPerspective(frame,H,(frame.shape[1],frame.shape[0]))
        right_frame = right_frame[:, int(distance) * (-1):]

        left_frame = img_add(left_frame,right_frame)

        # left_frame = old_frame.copy()
        distance = 0
        distance_row = 0
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    img = cv2.add(frame,mask)

    canny_img = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # cv2.imshow('img',img)

    cv2.imshow('canny', canny_img)
    cv2.imshow('result',left_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('pin.jpg',left_frame)

        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()


cv2.destroyAllWindows()
cap.release()