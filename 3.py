# -*- coding=utf-8 -*-
import socket
import threading
import sys
import os
import struct

import cv2
import numpy as np

def deal_data(conn, addr):
    print('Accept new connection from {0}'.format(addr))
    image1 = []
    while True:
        fileinfo_size = struct.calcsize('128sl')  # linux 和 windows 互传 128sl 改为 128sq  机器位数不一样，一个32位一个64位
        buf = conn.recv(fileinfo_size)
        data = buf
        print('收到的字节流：', buf, type(buf))
        # print('data,', data)
        # data = base64.b64decode(data)
        if not data or len(data) == 0:
            break
        image1.extend(data)
        # print("get over")
        image = np.asarray(bytearray(image1), dtype="uint8")
        # print("1", image)
        # print("2",len(image))  39559
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        conn.close()
        break


def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 12345)) # 这里换上自己的ip和端口
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print("Waiting...")
    while True:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()


if __name__ == '__main__':
    socket_service()

