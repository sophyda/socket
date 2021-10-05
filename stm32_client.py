import socket
import sys
import serial
#
# port = 3
# ser = serial.Serial('COM3', 9600, timeout=2)
#

def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.43.20', 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024))  # 目的在于接受：Accept new connection from (...
    while 1:
        # data = input('please input work: ').encode()
        # data_read = ser.read()
        data_read = 't27.12b56.22b1000b1000b1000b30.47755b114.8585'.encode()
        s.send(data_read)
        print(s.recv(1024))
        if s.recv(1024).decode()=='0':
            print(1)
        if data_read == 'exit':
            break
    s.close()


if __name__ == '__main__':
    socket_client()