import cv2
from lib_ip.ip_preprocessing import *


def nothing():
    pass


input_path_img = 'data\\input\\353.jpg'
org, grey = read_img(input_path_img, 800)

cv2.namedWindow('grad')
cv2.createTrackbar('grad', 'grad', 4, 100, nothing)
while True:
    grad = cv2.getTrackbarPos('grad', 'grad')
    bin = binarization(org, grad)
    cv2.imshow('bin', bin)
    cv2.waitKey(10)