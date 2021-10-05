from socket import *
#返回时间的
from time import ctime
#定义域名和端口号
HOST,PORT='',555
#创建缓冲区的大小(1M)
BUFFER_SIZE=1024
ADDR=(HOST,PORT)
#创建服务器的套接字 第一个参数代表IP 第二个基于TCP的数据流 代表TCP/IP协议的编程
tcpServerSocket = socket(AF_INET,SOCK_STREAM)
#绑定域名和端口号
tcpServerSocket.bind(ADDR)
#监听连接，传入连接请求的最大数
tcpServerSocket.listen(6)

#等待客户的连接
while True:
    #客户端的连接对象
    tcpClientSocket,addr = tcpServerSocket.accept()
    print('连接服务器的客户端对象：',addr)
    #定义一个循环开始接受数据
    while True:
        #请求是连接到客户端TCP套接字，接受缓冲区读取字节的长度    decode()解码 bytes --> str
        data = tcpClientSocket.recv(BUFFER_SIZE).decode()
        if not data:
            break
        print('data=',data)
        #发送的时间信息
        tcpClientSocket.send(('[%s] %s'%(ctime(),data)).encode())
    # 关闭资源
    tcpClientSocket.close()
tcpServerSocket.close()
