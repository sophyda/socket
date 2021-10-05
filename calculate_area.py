import ctypes
import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ctypes import *
import sys

sys.path.append("./detection")
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import cv2
import cv2_util

import random
import time
import datetime


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)

class calculate(object):
    def __init__(self):
        self.dll = ctypes.cdll.LoadLibrary(r'D:\VScode\trangle_area\x64\Release\trangle_area.dll')
        self.dll.ret_area.restype = c_float

        self.device = torch.device('cpu')
        num_classes = 2
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        self.model.to(self.device)
        self.model.eval()
        save = torch.load('model.pth')
        self.model.load_state_dict(save['model'])

        '''
        动态库
        '''

        # area = dll.construct_pcd(op_img.shape[0], op_img.shape[1],
        #                          op_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    def bool(self,mask,img):
        img_new = img.copy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] == 0:
                    img_new[i,j] =0
        return img_new

    def random_color(self):
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)

        return (b, g, r)


    def toTensor(self,img):
        assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img.float().div(255)  # 255也可以改为256


    def predict_img_dep(self, img, dep, photoid):
        # img, _ = dataset_test[0]
        img = cv2.imread(img)
        dep = cv2.imread(dep,-1)
        img_copy = img.copy()
        result = img.copy()
        dst = img.copy()
        img = self.toTensor(img)

        names = {'0': 'background', '1': 'ye'}
        # put the model in evaluati
        # on mode

        prediction = self.model([img.to(self.device)])

        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']
        masks = prediction[0]['masks']
        area_list = []
        # print(np.where(scores>0.9)[0])
        dst1 = None
        for idx in range(boxes.shape[0]):
            if scores[idx] >= 0.7:

                mask = masks[idx, 0].mul(255).byte().cpu().numpy()

                #
                # cv2.imshow('',masks)
                # cv2.waitKey()
                # print(mask[200,200])
                color = random_color()
                thresh = mask
                contours, hierarchy = cv2_util.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(dst, contours, -1, color, -1)

                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                name = names.get(str(labels[idx].item()))
                # cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=2)
                # cv2.putText(result, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

                dst1 = cv2.addWeighted(result, 0.7, dst, 0.5, 0)  # 0.7 0.5
                # i += 1

                # img_pro = self.bool(mask,img_copy)

                img_pro = np.array(mask)

                img_pro = img_pro.reshape(img_pro.shape[0],img_pro.shape[1], 1)
                area = self.dll.ret_area(img_pro.shape[0], img_pro.shape[1],
                                         img_pro.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                         dep.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
                area_list.append(area*10000)
        cv2.imwrite('mask/m_'+str(photoid)+'.jpg',dst1)
        return area_list


    def predict_img(self,img):
        # img, _ = dataset_test[0]
        # img = cv2.imread(image)
        img = cv2.resize(img, (512, 512))
        result = img.copy()
        dst = img.copy()
        img = self.toTensor(img)

        names = {'0': 'background', '1': 'ye'}
        # put the model in evaluati
        # on mode

        prediction = self.model([img.to(self.device)])

        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']
        masks = prediction[0]['masks']

        i=0
        for idx in range(boxes.shape[0]):
            if scores[idx] >= 0.4:
                i+=1


if __name__ == '__main__':
    a = calculate()
    from centernet import CenterNet
    # centernet = CenterNet()
    for i in range(0,20):
        imgf = 'robot_img/'+str(i)+'.jpg'
        depf = 'robot_img/'+str(i)+'.png'

        area= a.predict_img_dep(imgf, depf, i+1)
        area = sorted(area)[2:-2]

        area = np.mean(area)

        # image = Image.open(imgf)
        # image = image.convert("RGB")
        # r_image, height, yps = centernet.detect_image(image)
        # r_image.save('mask/s'+str(i)+'.jpg')



        print(area)

