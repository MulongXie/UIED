import cv2
from os.path import join as pjoin
import time
import json
import numpy as np
from random import randint as rint

import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_segment as seg
import detect_compo.lib_ip.file_utils as file
import detect_compo.lib_ip.block_division as blk
import detect_compo.lib_ip.Component as Compo
from config.CONFIG_UIED import Config


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


input_path_img = '../data/input/0.jpg'
output_root = 'data/output'
resized_height = resize_height_by_longest_edge(input_path_img)

org, grey = pre.read_img(input_path_img, resized_height)
binary = pre.binarization(org, grad_min=4, show=True)

# blk.block_division(grey, org, 3, show=True, write_path='blk.png')


_, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
draw_cnt = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
draw_appx = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
draw_hull = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    cnt = contours[i]
    if cv2.arcLength(cnt, True) < 20:
        continue
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    hull = cv2.convexHull(approx)
    cv2.drawContours(draw_cnt, contours, i, (rint(0,255), rint(0,255), rint(0,255)))
    cv2.drawContours(draw_appx, [approx], 0, (rint(0,255), rint(0,255), rint(0,255)))
    cv2.drawContours(draw_hull, [hull], 0, (rint(0,255), rint(0,255), rint(0,255)))

cv2.imshow('contour', draw_cnt)
cv2.imshow('approximate', draw_appx)
cv2.imshow('hull', draw_hull)
cv2.waitKey()