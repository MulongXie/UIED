import cv2
import numpy as np
from random import randint as rint
from config.CONFIG_UIED import Config


C = Config()


def draw_bounding_box_class(org, corners, classes, color_map=C.COLOR, line=2,
                            draw_text=False, show=False, write_path=None):
    """
    Draw bounding box of components with their classes on the original image
    :param org: original image
    :param corners: [(top_left, bottom_right)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color_map: colors mapping to different components
    :param line: line thickness
    :param compo_class: classes matching the corners of components
    :param show: show or not
    :return: labeled image
    """
    board = org.copy()
    for i in range(len(corners)):
        # if not draw_text and classes[i] == 'text':
        #     continue
        board = cv2.rectangle(board, corners[i][0], corners[i][1], color_map[classes[i]], line)
        board = cv2.putText(board, classes[i], (corners[i][0][0]+5, corners[i][0][1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[classes[i]], 2)
    if show:
        cv2.imshow('a', board)
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def draw_bounding_box(org, corners, color=(0, 255, 0), line=2, show=False, write_path=None):
    """
    Draw bounding box of components on the original image
    :param org: original image
    :param corners: [(top_left, bottom_right)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color: line color
    :param line: line thickness
    :param show: show or not
    :return: labeled image
    """
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, corners[i][0], corners[i][1], color, line)
    if show:
        cv2.imshow('a', board)
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def draw_line(org, lines, color=(0, 255, 0), show=False):
    """
    Draw detected lines on the original image
    :param org: original image
    :param lines: [line_h, line_v]
            -> line_h: horizontal {'head':(column_min, row), 'end':(column_max, row), 'thickness':int)
            -> line_v: vertical {'head':(column, row_min), 'end':(column, row_max), 'thickness':int}
    :param color: drawn color
    :param show: show or not
    :return: image with lines drawn
    """
    board = org.copy()
    line_h, line_v = lines
    for line in line_h:
        cv2.line(board, tuple(line['head']), tuple(line['end']), color, line['thickness'])
    for line in line_v:
        cv2.line(board, tuple(line['head']), tuple(line['end']), color, line['thickness'])
    if show:
        cv2.imshow('img', board)
        cv2.waitKey(0)
    return board


def draw_boundary(boundaries, shape, show=False):
    """
    Draw boundary of objects on the black withe
    :param boundaries: boundary: [top, bottom, left, right]
                        -> up, bottom: (column_index, min/max row border)
                        -> left, right: (row_index, min/max column border) detect range of each row
    :param shape: shape or original image
    :param show: show or not
    :return: drawn board
    """
    board = np.zeros(shape[:2], dtype=np.uint8)  # binary board

    for boundary in boundaries:
        # up and bottom: (column_index, min/max row border)
        for point in boundary[0] + boundary[1]:
            board[point[1], point[0]] = 255
        # left, right: (row_index, min/max column border)
        for point in boundary[2] + boundary[3]:
            board[point[0], point[1]] = 255
    if show:
        cv2.imshow('rec', board)
        cv2.waitKey(0)
    return board


def draw_region(region, broad, show=False):
    color = (rint(0,255), rint(0,255), rint(0,255))
    for point in region:
        broad[point[0], point[1]] = color

    if show:
        cv2.imshow('region', broad)
        cv2.waitKey()
    return broad


def draw_region_bin(region, broad, show=False):
    for point in region:
        broad[point[0], point[1]] = 255

    if show:
        cv2.imshow('region', broad)
        cv2.waitKey()
    return broad
