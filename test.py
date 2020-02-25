import cv2
import numpy as np
import lib_ip.ip_preprocessing as pre
import lib_ip.ip_detection as det


def nothing(x):
    pass


def get_contour(org, binary):
    def cvt_bbox(bbox):
        '''
        x,y,w,h -> colmin, rowmin, colmax, rowmax
        '''
        return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    board = org.copy()
    hie, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_contour = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 200:
            continue
        cnt = cv2.approxPolyDP(contours[i], 0.001*cv2.arcLength(contours[i], True), True)
        res_contour.append(cnt)
    cv2.drawContours(board, res_contour, -1, (0,0,255), 1)
    return board


img_file = 'E:\\Mulong\\Datasets\\rico\\combined\\17.jpg'
resize_height = 800

cv2.namedWindow('control')
cv2.createTrackbar('resize_height', 'control', 800, 1600, nothing)
cv2.createTrackbar('grad_min', 'control', 4, 255, nothing)
cv2.createTrackbar('c1', 'control', 1, 1000, nothing)
cv2.createTrackbar('c2', 'control', 1, 1000, nothing)


while 1:
    resize_height = cv2.getTrackbarPos('resize_height', 'control')
    grad_min = cv2.getTrackbarPos('grad_min', 'control')
    c1 = cv2.getTrackbarPos('c1', 'control')
    c2 = cv2.getTrackbarPos('c2', 'control')

    org, grey = pre.read_img(img_file, resize_height)
    # org = cv2.medianBlur(org, 3)
    # org = cv2.GaussianBlur(org, (3,3), 0)
    binary = pre.preprocess(org, grad_min)
    # canny = cv2.Canny(grey, c1, c2)
    # hie, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # b_contour = get_contour(org, binary)
    # c_contour = get_contour(org, canny)
    # b_contour = cv2.hconcat([b_contour, c_contour])
    # binary = cv2.hconcat([binary, canny])

    # cv2.imshow('before', binary)
    # binary = cv2.medianBlur(binary, 3)

    # cv2.imshow('b_cnt', b_contour)
    cv2.imshow('org', org)
    cv2.imshow('bin', binary)
    # cv2.imshow('canny', canny)
    cv2.waitKey(10)
