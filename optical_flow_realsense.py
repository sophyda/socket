import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


def dep_add(img1, img2):
    # print(img1.shape,img2.shape)
    img_new = np.ndarray([img1.shape[0], img1.shape[1] + img2.shape[1]],dtype=np.uint16)
    img_new[:, 0:img1.shape[1]] = img1
    img_new[:, img1.shape[1]:] = img2
    return np.array(img_new)


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
frames = pipeline.wait_for_frames()
# frames.get_depth_frame() is a 640x360 depth image

# Align the depth frame to color frame
aligned_frames = align.process(frames)

# Get aligned frames
aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
color_frame = aligned_frames.get_color_frame()

depth_image = np.asanyarray(aligned_depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())
old_frame = color_image

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
left_frame = old_frame.copy()
left_dep = depth_image.copy()

point_per = []
distance = 0
distance_per = 0

while(1):

    # ret,frame = cap.read()
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    frame = color_image

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
    if distance > 2:
        right_frame = frame[:,int(distance)*(-1):]
        left_frame = img_add(left_frame,right_frame)

        right_dep = depth_image[:,int(distance)*(-1):]
        left_dep = dep_add(left_dep, right_dep)

        # left_frame = old_frame.copy()
        distance = 0
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    img = cv2.add(frame,mask)
    cv2.imshow('img',img)
    cv2.imshow('dep',left_dep)
    cv2.imshow('result',left_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('dep.png',left_dep)
        cv2.imwrite('rgb.jpg',left_frame)
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()


cv2.destroyAllWindows()
cap.release()