import cv2
import pyzbar.pyzbar as pyzbar


def decodeQR(image):
    barcodes = pyzbar.decode(image)
    if barcodes != []:
        print('成功检测二维码')
    for barcode in barcodes:
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)

        # 提取二维码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 10)

        # 向终端打印条形码数据和条形码类型
        # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return image


def detect():
    # 读取图片
    # frame = cv2.imread('1.jpg')
    # im = decodeQR(frame)
    # cv2.imshow("camera", im)
    # cv2.imwrite('zbar.jpg', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cap=cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        frame = decodeQR(frame)
        cv2.imshow("cap", frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()