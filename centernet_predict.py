from centernet import CenterNet
from PIL import Image
import os
import numpy as np
import glob
import cv2
def predict(filename):

    centernet = CenterNet()
    image = Image.open(filename)
    image = image.convert("RGB")
    r_image, height, yps = centernet.detect_image(image)
    r_image.save('re.jpg')
    print(yps)
    # r_image.show()
    return r_image, yps
predict('1s.jpg')

