# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png

# import the necessary packages
from panorama import Stitcher
import argparse

import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
# 	help="path to the first image")
# ap.add_argument("-s", "--second", required=True,
# 	help="path to the second image")
# args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread('./RGB/color1.png',-1)
imageB = cv2.imread('./RGB/color2.png',-1)
# dep1 = cv2.imread('./examples/1.png',-1)
# dep2 = cv2.imread('./examples/2.png',-1)
# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis, a2,result_dis) = stitcher.stitch([imageA, imageB], showMatches=True)
# r = cv2.warpPerspective(dep2, a2,
# 		(dep1.shape[1] + dep2.shape[1], dep2.shape[0]))
# r[0:dep1.shape[0], 0:dep1.shape[1]] = dep1

# cv2.imshow('',r)
# print(a2)
# cv2.imwrite('dep.png',r)
# cv2.imwrite('rgbD.jpg',result)
# # print(r.shape)
# cv2.imshow('',result_dis)
cv2.imshow("Result", vis)
#apply colormap on deoth image(image must be converted to 8-bit per pixel first)
# im_color=cv2.applyColorMap(cv2.convertScaleAbs(r,alpha=15),cv2.COLORMAP_JET)
#convert to mat png
# im=Image.fromarray(im_color)
#save image
# combined_image = cv2.addWeighted(result,0.7,im_color,0.3,0)
# cv2.imshow('',combined_image)
cv2.waitKey(0)

print(result.shape)
cv2.waitKey(0)