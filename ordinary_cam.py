import cv2

cap = cv2.VideoCapture(1)

while 1:
    ret, frame = cap.read()
    cv2.imshow('f',frame)
    key = cv2.waitKey(2)
    if key == 27:
        break