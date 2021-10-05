import socket
import cv2
import numpy
import time
import sys
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
device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
print(device_config)

# Start cameras using modified configuration
pyK4A.device_start_cameras(device_config)
def SendVideo():
    # 建立sock连接
    # address要连接的服务器IP地址和端口号
    address = ('192.168.1.100', 8002)
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

    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_PNG_STRATEGY_DEFAULT), 100]

    while True:
        # 停止0.1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
        time.sleep(0.01)
        # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
        # '.jpg'表示将图片按照jpg格式编码。

        pyK4A.device_get_capture()

        # Get the depth image from the capture
        depth_image_handle = pyK4A.capture_get_depth_image()

        # Get the color image from the capture
        color_image_handle = pyK4A.capture_get_color_image()

        # Check the image has been read correctly
        if depth_image_handle and color_image_handle:

            # Read and convert the image data to numpy array:
            color_image = pyK4A.image_convert_to_numpy(color_image_handle)[:, :, :3]

            # Transform the depth image to the color format
            transformed_depth_image = pyK4A.transform_depth_to_color(depth_image_handle, color_image_handle)

            # Convert depth image (mm) to color, the range needs to be reduced down to the range (0,255)
            transformed_depth_color_image = cv2.applyColorMap(np.round(transformed_depth_image / 30).astype(np.uint8),
                                                              cv2.COLORMAP_JET)

            # Add the depth image over the color image:
            combined_image = cv2.addWeighted(color_image, 0.7, transformed_depth_color_image, 0.3, 0)

            # Plot the image
            # cv2.namedWindow('Colorized Depth Image', cv2.WINDOW_NORMAL)
            # cv2.imshow('Colorized Depth Image', combined_image)
            # cv2.imshow('', transformed_depth_image)
            # k = cv2.waitKey(25)


            result, imgencode = cv2.imencode('.png', transformed_depth_image)
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
            if len(receive): print(str(receive, encoding='utf-8'))

            pyK4A.image_release(depth_image_handle)
            pyK4A.image_release(color_image_handle)
        # 读取下一帧图片

    sock.close()


if __name__ == '__main__':
    SendVideo()