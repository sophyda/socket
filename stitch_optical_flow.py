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
cap = cv2.VideoCapture(1)

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes


mask = np.zeros_like(old_frame)
img_canny = np.zeros((old_frame.shape[0], old_frame.shape[1]))

left_frame = old_frame.copy()
# point_per = []
distance = 0
distance_per = 0

distance_row_per = 0
distance_row = 0
# g3 = np.array(p0.shape)

turn = 0  # 轮次

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
    # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #
    #     point_per.append(abs(c-a))
    #     point_row_per.append(b-d)
    #
    #     ori_arr.append(b-d)    # 调试
    #         # print(c-a)
    #     # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # # print(np.array(point_per))
    #
    # '''
    # 在竖直方向使用ransac
    # '''
    # # point_row_per = np.array(point_row_per)
    # # print('per',point_row_per)
    # # point_row_per_y = [i+1 for i in range(len(point_row_per))]
    # # point_row_per_y = np.array(point_row_per_y)
    # # point_row_per_y = point_row_per_y.reshape(point_row_per_y.shape[0],1)
    # # model_x = ransac.LinearRegression(point_row_per_y, point_row_per)
    # # model_x.RANSAC(10,3,3)
    # # point_row_per = model_x.predict_result(point_row_per_y)
    # # print('ran',point_row_per)
    #
    # rans_arr = np.array(ori_arr)
    # rans_arr = np.append(rans_arr, point_row_per)
    #
    # distance_row_per = np.mean(point_row_per)
    # distance_row = distance_row + distance_row_per


    '''
    在水平方向使用ransac
    '''

    # point_per = np.array(point_per)
    # point_per_x = [i + 1 for i in range(len(point_per))]
    # point_per_x = np.array(point_per_x)
    # point_per_x = point_per_x.reshape(point_per_x.shape[0], 1)
    # model_x = ransac.LinearRegression(point_per_x, point_per)
    # model_x.RANSAC(20, 3, 3)
    # point_per = model_x.predict_result(point_per_x)
    #
    # rans_arr = np.array(rans_arr)
    # rans_arr = np.append(rans_arr, point_per)
    #
    # distance_per = np.mean(point_per)
    # distance = distance + abs(distance_per)
    #





    # print(len(point_per))

    p0 = good_new.reshape(-1,1,2)
    if distance > 1:

        # transformation_matrix = np.float32([[1, 0, 0], [0, 1,distance_row]])

        right_frame = frame[:,int(distance)*(-1):]

        # right_frame = cv2.warpAffine(right_frame, transformation_matrix, (right_frame.shape[1],right_frame.shape[0]))
        left_frame = img_add(left_frame, right_frame)

        # left_frame = old_frame.copy()
        distance = 0
        distance_row = 0
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    img_canny_ori = img_canny.copy()
    print(img_canny_ori.shape, img_canny_ori.shape)
    #img = cv2.add(frame,mask)


    img_canny = cv2.Canny(frame, 18, 100)
    old_gray = frame_gray.copy()

    img_sub = img_canny - img_canny_ori
    cv2.imshow('ori',img_canny_ori)
    cv2.imshow('img',img_canny )
    cv2.imshow('result',left_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('pinjie.jpg',left_frame)

        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    turn += 1

cv2.destroyAllWindows()
cap.release()



