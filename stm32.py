# import serial
# #
# # # 连接串口
# # serial = serial.Serial('COM3', 9600, timeout=2)  # 连接COM14,波特率位115200
# # if serial.isOpen():
# #     print('串口已打开')
# # else:
# #     print('串口未打开')
# #
# # # 关闭串口
# # serial.close()
# #
# # if serial.isOpen():
# #     print('串口未关闭')
# # else:
# #     print('串口已关闭')
import time

import serial

port = 3
ser = serial.Serial('COM3', 9600, timeout=2)

# print(ser.portstr)

# baudrate = 9600
# ser.setBaudrate(baudrate)
# ser.open()++++++++++++++++
# print(ser.isOpen())
a = ''
i = 0
start_save = 0
# print(ser)
while (1):
    i+=1
    # datainput = input("Please input the character:\n")
    n = ser.write(b'a\r\n')
    # n = ser.write(10)
    data = ser.read()
    data_rec = data.decode()
    a = a+str(data_rec)
    if data_rec == 't':
        # if len(a) == 16:
      print(a)
      print('-------')
      a=''



    #print(data.decode())
    # print(data_rec)
    # print('---------')

    # if data_rec != '\n' and data_rec !='\r':
    #     a.append(data_rec)
    #
    # if len(a) == 12:
    #     print(a)
    #     a = []



