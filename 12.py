# coding:utf-8
import socket
import sys
import time
import serial
import numpy
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import threading

# Create a pipeline

port = 3
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=2)
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
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


def senddep():
    address = ('192.168.43.20', 8004)
    try:
        # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    while True:
        # 停止0.1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
        time.sleep(0.01)
        # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
        # '.jpg'表示将图片按照jpg格式编码。

        # Get aligned frames

        # 建立图像读取对象
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Validate that both frames are valid
        result, imgencode = cv2.imencode('.png', depth_image)
        # 建立矩阵
        data = numpy.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        stringData = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        sock.send(str.encode(str(len(stringData)).ljust(16)));
        # 发送数据
        sock.send(stringData);
        # 读取服务器返回值
        receive = sock.recv(1024)
        # if len(receive): print(str(receive, encoding='utf-8'))
        # 读取下一帧图片

        if cv2.waitKey(10) == 27:
            break
    sock.close()


def stm32_socket():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.43.20', 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    # print(s.recv(1024))  # 目的在于接受：Accept new connection from (...
    while 1:
        # data = input('please input work: ').encode()
        data_read = None
        data_read = ser.read()
        if data_read == '':
            data_read = 't27.12b56.22b1000b1000b1000b30.47755b114.8585'.encode()
        # data_read = 't27.12b56.22b1000b1000b1000b30.47755b114.8585'.encode()
        s.send(data_read)
        order = s.recv(1024).decode()
        print(order)
        if order == 'a':
            ser.write(b'a\r\n')
        if order == 'b':
            print('---------------')
            ser.write(b'b\r\n')
        if order == 'c':
            ser.write(b'c\r\n')
        if order == 'd':
            ser.write(b'd\r\n')
        if data_read == 'exit':
            break

    s.close()


def SendVideo():
    # 建立sock连接
    # address要连接的服务器IP地址和端口号
    address = ('192.168.43.20', 8003)
    try:
        # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    while True:
        # 停止0.1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
        time.sleep(0.01)
        # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
        # '.jpg'表示将图片按照jpg格式编码。

        # Get aligned frames

        # 建立图像读取对象
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        # Validate that both frames are valid
        result, imgencode = cv2.imencode('.jpg', color_image, encode_param)
        # 建立矩阵
        data = numpy.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        stringData = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        sock.send(str.encode(str(len(stringData)).ljust(16)));
        # 发送数据
        sock.send(stringData);
        # 读取服务器返回值
        receive = sock.recv(1024)
        # if len(receive): print(str(receive, encoding='utf-8'))
        # 读取下一帧图片

        if cv2.waitKey(10) == 27:
            break
    sock.close()


def SendVideo_cam():
    # 建立sock连接
    # address要连接的服务器IP地址和端口号
    address = ('192.168.43.20', 8002)
    try:
        # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    # 建立图像读取对象
    capture = cv2.VideoCapture(0)
    # 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
    ret, frame = capture.read()
    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    while ret:
        # 停止0.1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
        time.sleep(0.01)
        # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
        # '.jpg'表示将图片按照jpg格式编码。
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        # 建立矩阵
        data = numpy.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        stringData = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        sock.send(str.encode(str(len(stringData)).ljust(16)));
        # 发送数据
        sock.send(stringData);
        # 读取服务器返回值
        receive = sock.recv(1024)
        # if len(receive): print(str(receive, encoding='utf-8'))
        # 读取下一帧图片
        ret, frame = capture.read()
        # if cv2.waitKey(10) == 27:
        #   break
    sock.close()


if __name__ == '__main__':
    t1 = threading.Thread(target=SendVideo)
    t2 = threading.Thread(target=SendVideo_cam)
    t3 = threading.Thread(target=stm32_socket)
    t4 = threading.Thread(target=senddep)
    t4.start()
    t1.start()
    t2.start()
    t3.start()