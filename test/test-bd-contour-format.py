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


def bd_reformat(compo_bd):
    new_bd = []
    for point in compo_bd[0:2]:
        new_bd.append(point)
    for point in compo_bd[2:4]:
        new_bd.append([point[1], point[0]])
    return new_bd


input_path_img = '../data/input/0.jpg'
output_root = 'data/output'
resized_height = resize_height_by_longest_edge(input_path_img)

org, grey = pre.read_img(input_path_img, resized_height)
binary = pre.binarization(org, grad_min=4, show=True)

# blk.block_division(grey, org, 3, show=True, write_path='blk.png')
uicompos = det.component_detection(binary, min_obj_area=50)
draw.draw_boundary(uicompos, org.shape, 'boundary', show=True)
draw.draw_boundary(uicompos, org.shape, 'boundary_closed', window_name='c', show=True)

drawing = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
for compo in uicompos:
    cv2.drawContours(drawing, [compo.boundary_closed], 0, (rint(0,255), rint(0,255), rint(0,255)))
cv2.imshow('b', drawing)
cv2.waitKey()