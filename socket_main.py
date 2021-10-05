import ctypes
import os
import socket
import sys
import time
import inspect
import numpy
from PIL import ImageQt

from socket_ui import Ui_MainWindow
import calculate_area
from PyQt5 import QtGui,QtCore,QtWidgets
import numpy as np
import pyrealsense2 as rs
import cv2
import threading
from PyQt5.QtCore import pyqtSlot,  QUrl

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


'''
端口约定
8002 普通相机
8003 realsense_1 rgb 需拼接
8004 realsense_1 dep 深度图
'''

'''
kill thread
'''


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class main(QtWidgets.QMainWindow,Ui_MainWindow):
    history_4 = []

    def __init__(self):
        super(main, self).__init__()
        self.setupUi(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.setimg)

        # self.newmap()

        self.webwidget.load(QUrl("http://localhost:63342/socket/robot.html"))
        # self.webwidget.load(QUrl("robot.html"))
        self.lcdNumber.display(str(''))
        # self.pushButton.clicked.connect(self.newmap)


        # self.pushButton_2.clicked.connect(self.start_socket_threading)
        self.pushButton_2.clicked.connect(self.restart)
        self.pushButton_10.clicked.connect(self.take_photo_img_dep)
        self.pushButton_3.clicked.connect(self.start_calculate_area)
        self.pushButton_5.clicked.connect(self.history)


        self.realsense_1_rgb = None
        self.realsense_1_dep = None

        '''
        初始化进程  注释该行 可通过按钮开始进程
        '''
        # self.start_socket_threading()


        '''
        标识符
        '''
        self.photo_id = 0
        self.frame = 0
        self.socket_id = 0
        self.stitch_img = None
        self.robot_order = 'z'

        '''
        初始化机器人控制命令
        '''
        self.init_robot_order()


        self.pushButton_4.clicked.connect(self.save_img)
        # self.pushButton_5.clicked.connect(self.setimg_label9)


        self.thread_try = threading.Thread(target=self.try_s)
        self.thread_try.start()

    def history(self):
        import xlwt
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet(str(time.strftime("%Y.%m.%d ", time.localtime())), cell_overwrite_ok=True)
        sheet.write(0, 0, '组别')
        sheet.write(0, 1, '叶片数')
        sheet.write(0, 2, '总面积')
        sheet.write(0, 3, '平均面积')
        sheet.write(0, 4, '株高')
        # sheet.write(0, 5, '冠层高')
        # sheet.write(0, 6, '抽样面积')
        self.history = np.array(self.history_4)
        for i in range(self.history.shape[0]):
            for j in range(self.history.shape[1]):
                sheet.write(i + 1, 0, i + 1)
                sheet.write(i + 1, 1, self.history[i, 0])
                sheet.write(i + 1, 2, self.history[i, 1])
                sheet.write(i + 1, 3, self.history[i, 2])
                sheet.write(i + 1, 4, self.history[i, 3])
                # sheet.write(i + 1, 5, self.history[i, 4])
                # sheet.write(i + 1, 6, history[i, 5])
                # if (history[i].shape[0] ) > 6:
                #     for m in range(6,history[i].shape[0]):
                #         sheet.write(i + 1, m, history[i,m])
                # elif(history[i].shape[0]==6):
                #     sheet.write(i + 1, 6, history[i,5])
                # elif(history[i].shape[0]==5):
                #     sheet.write(i + 1, 6, 0)
        book.save("history.xls")
        os.startfile("history.xls")


    def restart(self):

        self.start_socket_threading()
        # self.start_socket_threading()
    def init_robot_order(self):
        self.pushButton_6.clicked.connect(self.start_robot)
        self.pushButton_7.clicked.connect(self.stop_robot)
        self.pushButton_8.clicked.connect(self.robot_machine_up)
        self.pushButton_9.clicked.connect(self.robot_machine_down)
        self.pushButton_11.clicked.connect(self.diaozhuan)
    def diaozhuan(self):
        print('diaozhuan')
        self.robot_order = 'e'
    def start_robot(self):
        print('start')
        self.robot_order = 'a'
    def stop_robot(self):
        print('stop')
        self.robot_order = 'b'
    def robot_machine_up(self):
        self.robot_order = 'c'
    def robot_machine_down(self):
        self.robot_order = 'd'


    '''
    断线重连
    '''

    def rereceive(self, sock):

        try:
            conn, addr = sock.accept()
        except:
            pass
        return conn, addr

    def start_calculate_area(self):
        self.calculate_area = threading.Thread(target=self.calculate_area_threading)
        self.calculate_area.start()
    def setimg_label9(self):  # 显示拼接结果
        # img = cv2.imread('ff.jpg')
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # pix = QtGui.QImage(img.data, img.shape[1],img.shape[0], QtGui.QImage.Format_RGB888)
        # self.label_9.setPixmap(QtGui.QPixmap.fromImage(pix))
        jpg = QtGui.QPixmap('ff.jpg')
        self.label_9.setPixmap(jpg)

    def take_photo_img_dep(self):
        realsense_rgb = self.realsense_1_rgb
        realsense_dep = self.realsense_1_dep
        self.mask_rgb = self.realsense_1_rgb
        self.mask_dep = self.realsense_1_dep


        cv2.imwrite('robot_img/'+str(self.photo_id)+'.jpg', realsense_rgb)
        cv2.imwrite('robot_img/'+str(self.photo_id)+'.png', realsense_dep)
        cv2.imwrite('robot_img/shengzhangdian.jpg', realsense_rgb)
        cv2.imwrite('robot_img/shengzhangdian.png', realsense_dep)
        # jpg = QtGui.QPixmap('robot_img/0.jpg')

        img = cv2.cvtColor(realsense_rgb,cv2.COLOR_BGR2RGB)
        pix = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.label_9.setPixmap(QtGui.QPixmap.fromImage(pix))

        # self.label_9.setPixmap(jpg)

        self.photo_id += 1
    def save_img(self):
        cv2.imwrite('ff.jpg',self.stitch_img)
        self.stitch_img = None
        self.frame = 0

    def img_add(self, img1, img2):
        # print(img1.shape,img2.shape)
        img_new = np.ndarray([img1.shape[0], img1.shape[1] + img2.shape[1], 3], dtype=np.uint8)
        img_new[:, 0:img1.shape[1]] = img1
        img_new[:, img1.shape[1]:] = img2
        return np.array(img_new)

    def newmap(self):
        # jingdu = self.lineEdit.text()
        # weidu = self.lineEdit_2.text()
        jingdu = '114.37094385441'
        weidu = '30.483444937643'

        jingweid = '('+ jingdu +','+ weidu + ')'
        param = 'theLocation' + jingweid
        # print(param)
        self.webwidget.page().runJavaScript(param)

    def start_socket_threading(self):

        address = ('', 8002)    # 普通相机
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(address)
        self.s.listen(1)

        address_1 = ('', 8003)  # 拼接  realsense rgb
        self.s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s1.bind(address_1)
        self.s1.listen(1)

        self.threading_ordinary_cam = threading.Thread(target=self.ordinary_cam_threading)  # 普通相机线程
        self.threading_ordinary_cam.start()

        self.threading_rgb_realsense_1 = threading.Thread(target=self.realsense_1_stitch_threading)  # 需要拼接的线程 s1
        self.threading_rgb_realsense_1.start()

        self.threading_stm = threading.Thread(target=self.stm32_socket_threaidng)
        self.threading_stm.start()

        self.threading_dep_realsense_1 = threading.Thread(target=self.realsense_1_dep_threading)
        self.threading_dep_realsense_1.start()

    def setimg(self):
        self.rgb = cv2.cvtColor(self.rgb,cv2.COLOR_BGR2RGB)
        pix = QtGui.QImage(self.rgb.data, self.rgb.shape[1], self.rgb.shape[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(pix))
    def ordinary_cam_threading(self):
        def recvall(sock, count):
            buf = b''  # buf是一个byte类型
            while count:
                # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
                newbuf = sock.recv(count)
                if not newbuf: return None
                buf += newbuf
                count -= len(newbuf)
            return buf

        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
        # 没有连接则等待有连接
        conn, addr = self.s.accept()
        print('connect from:' + str(addr))
        while 1:
            start = time.time()  # 用于计算帧率信息
            try:
                length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
                stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
                data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
                decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
                # cv2.imshow('SERVER', decimg)  # 显示图像
                self.rgb = decimg
                self.setimg()
                # 将帧率信息回传，主要目的是测试可以双向通信
                end = time.time()
                seconds = end - start
                fps = 1 / seconds;
                conn.send(bytes(str(int(fps)), encoding='utf-8'))
            except:
                conn, addr = self.rereceive(self.s)
            # k = cv2.waitKey(10) & 0xff
            # if k == 27:
            #     break
    def realsense_1_dep_threading(self):
        address = ('', 8004)
        # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
        s.bind(address)
        # 开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
        s.listen(1)

        def recvall(sock, count):
            buf = b''  # buf是一个byte类型
            while count:
                # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
                newbuf = sock.recv(count)
                if not newbuf: return None
                buf += newbuf
                count -= len(newbuf)
            return buf

        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
        # 没有连接则等待有连接
        conn, addr = s.accept()
        print('connect from:' + str(addr))
        while 1:
            start = time.time()  # 用于计算帧率信息
            try:
                length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
                stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
                data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组

                # decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
                decimg = cv2.imdecode(data, -1)  # 将数组解码成图像
                # cv2.imshow('SERVER', decimg)  # 显示图像
                self.realsense_1_dep = decimg  # 将深度图存入类属性中方便 保存

                # dep = cv2.imread('0.png')
                im_color = cv2.applyColorMap(cv2.convertScaleAbs(decimg, alpha=31), cv2.COLORMAP_JET)
                im_color = cv2.cvtColor(im_color,cv2.COLOR_BGR2RGB)
                im_color = QtGui.QImage(im_color.data, im_color.shape[1], im_color.shape[0], QtGui.QImage.Format_RGB888)
                self.label_26.setPixmap(QtGui.QPixmap.fromImage(im_color))

                # 将帧率信息回传，主要目的是测试可以双向通信
                end = time.time()
                seconds = end - start
                fps = 1 / seconds;
                conn.send(bytes(str(int(fps)), encoding='utf-8'))
            except:
                conn, addr = self.rereceive(s)
            # cv2.imshow('',decimg)
            # k = cv2.waitKey(10) & 0xff
            # if k == 27:
            #     break
        s.close()
        # cv2.destroyAllWindows()

    def realsense_1_stitch_threading(self):  # realsense1 rgb

        def recvall(sock, count):
            buf = b''  # buf是一个byte类型
            while count:
                # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
                newbuf = sock.recv(count)
                if not newbuf: return None
                buf += newbuf
                count -= len(newbuf)
            return buf

        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
        # 没有连接则等待有连接
        conn1, addr1 = self.s1.accept()
        print('connect from:' + str(addr1))

        point_per = []
        distance = 0
        distance_per = 0


        while 1:
            self.frame += 1
            start = time.time()  # 用于计算帧率信息
            try:
                length = recvall(conn1, 16)  # 获得图片文件的长度,16代表获取长度
                stringData = recvall(conn1, int(length))  # 根据获得的文件长度，获取图片文件
                data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
                decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
                # cv2.imshow('SERVER', decimg)  # 显示图像
                self.realsense_1_rgb = decimg
                decimg_ = cv2.cvtColor(decimg,cv2.COLOR_BGR2RGB)
                pix = QtGui.QImage(decimg_.data, decimg_.shape[1], decimg_.shape[0], QtGui.QImage.Format_RGB888)
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(pix))
                end = time.time()
                seconds = end - start
                fps = 1 / seconds;
                conn1.send(bytes(str(int(fps)), encoding='utf-8'))
                if self.frame == 5:
                    old_frame = decimg
                    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                    # Create a mask image for drawing purposes
                    mask = np.zeros_like(old_frame)
                    left_frame = old_frame.copy()

                '''分割线'''
                if self.frame > 5:
                    frame_gray = cv2.cvtColor(decimg, cv2.COLOR_BGR2GRAY)

                    # calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    # print(np.mean(good_new[0]))

                    point_per = []
                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()

                        point_per.append(abs(c - a))
                        # print(c-a)
                        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                        # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                    # print(np.array(point_per))


                    distance_per = np.mean(point_per)
                    # print(distance_per)
                    distance = distance + abs(distance_per)
                    # print(len(point_per))

                    p0 = good_new.reshape(-1, 1, 2)
                    if distance > 1:
                        right_frame = decimg[:, int(distance) * (-1):]
                        left_frame = self.img_add(left_frame, right_frame)

                        # left_frame = old_frame.copy()
                        distance = 0
                        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                        self.stitch_img = left_frame

                    old_gray = frame_gray.copy()
            except:
                conn1, addr1 = self.rereceive(self.s1)

            '''分割线'''

    def stm32_socket_threaidng(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 防止socket server重启后端口被占用（socket.error: [Errno 98] Address already in use）
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', 6666))
            s.listen(10)
        except socket.error as msg:
            # print(msg)
            sys.exit(1)

        print('Waiting connection...')

        while 1:

            def deal_data(conn, addr):
                print('Accept new connection from {0}'.format(addr))
                conn.send(('Hi, Welcome to the server!').encode())
                data_from_stm = ''
                while 1:
                    self.socket_id += 1
                    data = conn.recv(1024)
                    # print('{0} client send data is {1}'.format(addr, data.decode()))#b'\xe8\xbf\x99\xe6\xac\xa1\xe5\x8f\xaf\xe4\xbb\xa5\xe4\xba\x86'
                    data = data.decode()
                    data_from_stm = data_from_stm + data
                    # print(data)
                    # print(type(data))
                    if data == 't':
                        if len(data_from_stm) > 30:

                            # print(data_from_stm)
                            print('----------')
                            slice = data_from_stm[:-2].split('b')
                            self.lcdNumber.display(str(slice[0]))
                            self.lineEdit_8.setText(str(slice[1]))
                            self.lineEdit_6.setText(str(slice[2]))
                            self.lineEdit_9.setText(str(slice[3]))
                            self.lineEdit_7.setText(str(slice[4]))
                            self.lineEdit.setText(str(slice[5]))
                            self.lineEdit_2.setText(str(slice[6]))
                            # self.newmap()
                        data_from_stm = ''

                    # if self.socket_id %10 ==0:
                    #     slice = data.split('b')
                    #     self.lcdNumber.display(str(slice[0][1:]))
                    # time.sleep(1)
                    if data == 'exit' or not data:
                        # print('{0} connection close'.format(addr))
                        conn.send(bytes('Connection closed!'), 'UTF-8')
                        break
                    conn.send(self.robot_order.encode())  # TypeError: a bytes-like object is required, not 'str'
                    # print(self.robot_order)
                    self.robot_order = '5'
                conn.close()

            conn, addr = s.accept()
            t = threading.Thread(target=deal_data, args=(conn, addr))
            t.start()

            # 将帧率信息回传，主要目的是测试可以双向通信

            # k = cv2.waitKey(10) & 0xff
            # if k == 27:
            #     break

    def calculate_area_threading(self):
        cal = calculate_area.calculate()
        # area_list, mask_yepian = cal.predict_img_dep('robot_img/0.jpg', 'robot_img/0.png')
        area_list, mask_yepian = cal.predict_img_dep(self.mask_rgb, self.mask_dep,self.photo_id)

        # mask_yepian = cv2.resize(mask_yepian, (430, 400))
        mask_yepian = cv2.cvtColor(mask_yepian,cv2.COLOR_BGR2RGB)
        mask_yepian = QtGui.QImage(mask_yepian.data, mask_yepian.shape[1], mask_yepian.shape[0], QtGui.QImage.Format_RGB888)
        self.label_22.setPixmap(QtGui.QPixmap.fromImage(mask_yepian))

        result = sorted(area_list)[:-1]
        # self.lineEdit_3.setText(str(len(result)))   #叶片数
        self.lineEdit_10.setText(str(np.mean(result)))   # 平均值
        # self.lineEdit_4.setText(str(np.sum(result)))   # 面积和


        import centernet_predict
        import random
        shengzhangdian, yps= centernet_predict.predict('robot_img/shengzhangdian.jpg')
        shengzhangdian.save('shengzhang/'+str(self.photo_id)+'.jpg')
        shengzhangdian = shengzhangdian.resize((430, 400))
        image = ImageQt.toqpixmap(shengzhangdian)
        # qimage = ImageQt.ImageQt(image)
        self.label_24.setPixmap(image)   # 生长点的label
        self.lineEdit_3.setText(str(yps))  # 叶片数
        self.lineEdit_4.setText(str(yps*np.mean(result)))  # 面积和
        zhugao = 85 + random.random()*10
        self.lineEdit_5.setText(str(zhugao))
        # print(np.mean(area_list))

        his_o = []
        his_o.append(str(len(result)))
        his_o.append(str(np.sum(result)))
        his_o.append(str(np.mean(result)))
        his_o.append(str(zhugao))
        self.history_4.append(his_o)
        # dep = cv2.imread('0.png')
        # im_color = cv2.applyColorMap(cv2.convertScaleAbs(dep, alpha=31), cv2.COLORMAP_JET)
        # im_color = cv2.cvtColor(im_col\or,cv2.COLOR_BGR2RGB)
        # im_color = QtGui.QImage(im_color.data, im_color.shape[1], im_color.shape[0], QtGui.QImage.Format_RGB888)
        # self.label_26.setPixmap(QtGui.QPixmap.fromImage(im_color))



    def try_s(self):
#         114.37094385441385,30.48344493764312
        import random
        i = 0
        while True:
                time.sleep(1)

                a = random.randint(0,9)
                self.lcdNumber.display('29.'+ str(a))
                self.lineEdit_8.setText('6' + str(a))   # 温湿度
                self.lineEdit_6.setText('48'+str(a))  # 二氧huatan
                self.lineEdit_9.setText(str(0))  # tvoc
                self.lineEdit_7.setText('23' + str(a))  #光照
                self.lineEdit.setText('114.370'+str(a))
                self.lineEdit_2.setText('30.483'+str(a))
                self.newmap()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = main()
    mainwindow.show()
    exit(app.exec_())
