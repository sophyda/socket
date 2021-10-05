import socketserver
import datetime
import base64

import numpy
import numpy as np
import cv2
import predict

predictimg = predict.predict()

class MyTCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        # print("get....")
        image1 = []
        try:
            while True:
                data = self.request.recv(10240)  # 拿到客户端发送的数据
                # print('data,', data)
                # data = base64.b64decode(data)
                if not data or len(data) == 0:
                    break
                image1.extend(data)
            # print("get over")
            image = np.asarray(bytearray(image1), dtype="uint8")
            # print("1", image)
            # print("2",len(image))  39559
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)



            img = predictimg.PredictImg(img)

            img = cv2.resize(img,(512,512))
            cv2.imshow('',img)
            cv2.waitKey()
            cv2.imwrite('1.jpg',img)
            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            #
            # result, imgencode = cv2.imencode('.jpg', img, encode_param)
            #
            # data = numpy.array(imgencode)
            #
            # stringData = data.tobytes()



            # cv2.imwrite("./1.jpg", image)
            # print("接收完成")
            cv2.waitKey()
        except Exception:
            print(self.client_address, "连接断开")
        finally:
            self.request.close()  # 异常之后，关闭连接

    # before handle,连接建立：
    def setup(self):
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time)
        print("连接建立：", self.client_address)

    # finish run  after handle
    def finish(self):
        print("释放连接")


if __name__ == "__main__":
    HOST, PORT = "", 5555
    # server=socketserver.TCPServer((HOST,PORT),MyTCPHandler)  #实例对象，传入参数

    # 多线程
    server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)
    server.serve_forever()  # 一直运行