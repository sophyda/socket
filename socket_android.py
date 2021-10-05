import socket
import serial


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "192.168.1.100"
port = 6768
print (host)
print (port)
serversocket.bind((host, port))
ser = serial.Serial('COM3', 9600, timeout=2)
serversocket.listen(10)
print ('server started and listening')
while 1:
    (clientsocket, address) = serversocket.accept()
    data = clientsocket.recv(1024).decode()
    if str(data) == 's\n':
        print(data)
        ser.write(b's')
    elif str(data) == 'p\n':
        print(data)
        ser.write(b'p')
    # if data != '00 00\n' and data != '0 0\n':
    #     # print(data)
    #     coordination = data.split(' ')
    #     print(coordination[0],'----',coordination[1])



# # 导入模块
# import socketserver
# import random
# # 定义一个类
# class MyServer(socketserver.BaseRequestHandler):
#   # 如果handle方法出现报错，则会进行跳过
#   # setup方法和finish方法无论如何都会进行执行
#   # 首先执行setup
#   def setup(self):
#     pass
#   # 然后执行handle
#   def handle(self):
#     # 定义连接变量
#     conn =self.request
#     # 发送消息定义
#     msg = "Hello World!"
#     # 消息发送
#     # conn.send(msg.encode())
#     # 进入循环，不断接收客户端消息
#     while True:
#       #接收客户端消息
#       data = conn.recv(1024)
#       # 打印消息
#       print(len(data.decode()))
#       # 接收到exit，则进行循环的退出
#       if data==b'exit':
#         break
#       # conn.send(data)
#       # conn.send(str(random.randint(1,1000)).encode())
#     conn.close()
#   # 最后执行finish
#   def finish(self):
#     pass
# if __name__=="__main__":
#   # 创建多线程实例
#   server = socketserver.ThreadingTCPServer(('192.168.1.100',6768),MyServer)
#   # 开启启动多线程，等待连接
#   server.serve_forever()

