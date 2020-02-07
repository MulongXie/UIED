import cv2
import numpy as np
from random import randint as rint
import time

import lib_ip.ip_preprocessing as pre
import lib_ip.ip_detection_utils as util
import lib_ip.ip_detection as det
import lib_ip.ip_draw as draw
import lib_ip.ip_segment as seg


def block_rectify(block_corner, components_corner):
    '''
    correct the coordinates of compos to the holistic image
    :param block_corner: corners of blocks
                        (top_left, bottom_right)
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    :param components_corner: list of corners of components needed to be corrected
                        [(top_left, bottom_right)]
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    :return:
    '''
    bias = block_corner[0]
    compos_corner_new = []
    for compo in components_corner:
        # column
        col_min = compo[0][0] + bias[0]
        col_max = compo[1][0] + bias[0]
        # row
        row_min = compo[0][1] + bias[1]
        row_max = compo[1][1] + bias[1]
        compos_corner_new.append(((col_min, row_min), (col_max, row_max)))

    return compos_corner_new


def block_erase(binary, blocks_corner, show=False, pad=2):
    '''
    erase the block parts from the binary map
    :param binary: binary map of original image
    :param blocks_corner: corners of detected layout block
    :param show: show or not
    :param pad: expand the bounding boxes of blocks
    :return: binary map without block parts
    '''

    bin_org = binary.copy()
    for block in blocks_corner:
        ((column_min, row_min), (column_max, row_max)) = block
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, binary.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, binary.shape[0])
        cv2.rectangle(binary, (column_min, row_min), (column_max, row_max), (0), -1)

    if show:
        cv2.imshow('before', bin_org)
        cv2.imshow('after', binary)
        cv2.waitKey()
    return binary


def block_is_top_or_bottom_bar(block, org_shape):
    height, width = org_shape[:2]
    ((column_min, row_min), (column_max, row_max)) = block
    if column_min < 5 and row_min < 5 and \
            width - column_max < 5 and row_max < 100:
        return True
    if column_min < 5 and height - row_min < 200 and \
            width - column_max < 5 and height - row_max < 5:
        return True
    return False


def block_division(grey, show=False, write_path=None):
    '''
    :param grey: grey-scale of original image
    :return: corners: list of [(top_left, bottom_right)]
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    '''

    def flood_fill_bfs(img, x_start, y_start, mark):
        '''
        Identify the connected region based on the background color
        :param img: grey-scale image
        :param x_start: row coordinate of start position
        :param y_start: column coordinate of start position
        :param mark: record passed points
        :return: region: list of connected points
        '''

        def neighbor(x, y):
            for i in range(x - 1, x + 2):
                if i < 0 or i >= img.shape[0]: continue
                for j in range(y - 1, y + 2):
                    if j < 0 or j >= img.shape[1]: continue
                    if mark[i, j] == 0 and abs(img[i, j] - img[x, y]) < 8:
                        stack.append([i, j])
                        mark[i, j] = 255

        stack = [[x_start, y_start]]  # points waiting for inspection
        region = [[x_start, y_start]]  # points of this connected region
        mark[x_start, y_start] = 255  # drawing broad
        while len(stack) > 0:
            point = stack.pop()
            region.append(point)
            neighbor(point[0], point[1])
        return region

    blocks = []
    mask = np.zeros((grey.shape[0], grey.shape[1]), dtype=np.uint8)
    broad = np.zeros((grey.shape[0], grey.shape[1], 3), dtype=np.uint8)

    row, column = grey.shape[0], grey.shape[1]
    for x in range(row):
        for y in range(column):
            if mask[x, y] == 0:
                region = flood_fill_bfs(grey, x, y, mask)
                # ignore small regions
                if len(region) < 500:
                    continue
                # get the boundary of this region
                boundary = util.boundary_get_boundary(region)
                # ignore lines
                if util.boundary_is_line(boundary, 5):
                    continue
                # ignore non-rectangle as blocks must be rectangular
                if not util.boundary_is_rectangle(boundary, 0.66, 0.25):
                    continue
                blocks.append(boundary)
                draw.draw_region(region, broad)
    if show:
        cv2.imshow('broad', broad)
        cv2.waitKey()
    if write_path is not None:
        cv2.imwrite(write_path, broad)

    blocks_corner = det.get_corner(blocks)
    return blocks_corner
