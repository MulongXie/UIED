import cv2
import numpy as np
from random import randint as rint
from config.CONFIG_UIED import Config


C = Config()


def draw_bounding_box_class(org, components, color_map=C.COLOR, line=2, show=False, write_path=None, name='board'):
    """
    Draw bounding box of components with their classes on the original image
    :param org: original image
    :param components: bbox [(column_min, row_min, column_max, row_max)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color_map: colors mapping to different components
    :param line: line thickness
    :param compo_class: classes matching the corners of components
    :param show: show or not
    :return: labeled image
    """
    board = org.copy()
    for compo in components:
        bbox = compo.put_bbox()
        board = cv2.rectangle(board, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_map[compo.category], line)
        # board = cv2.putText(board, compo.category, (bbox[0]+5, bbox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[compo.category], 2)
    if show:
        cv2.imshow(name, board)
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def draw_bounding_box(org, components, color=(0, 255, 0), line=2,
                      show=False, write_path=None, name='board', is_return=False, wait_key=0):
    """
    Draw bounding box of components on the original image
    :param org: original image
    :param components: bbox [(column_min, row_min, column_max, row_max)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color: line color
    :param line: line thickness
    :param show: show or not
    :return: labeled image
    """
    if not show and write_path is None and not is_return: return
    board = org.copy()
    for compo in components:
        bbox = compo.put_bbox()
        board = cv2.rectangle(board, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line)
    if show:
        cv2.imshow(name, board)
        if wait_key is not None:
            cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(name)
    if write_path is not None:
        # board = cv2.resize(board, (1080, 1920))
        # board = board[100:-110]
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


def draw_boundary(components, shape, show=False):
    """
    Draw boundary of objects on the black withe
    :param components: boundary: [top, bottom, left, right]
                        -> up, bottom: (column_index, min/max row border)
                        -> left, right: (row_index, min/max column border) detect range of each row
    :param shape: shape or original image
    :param show: show or not
    :return: drawn board
    """
    board = np.zeros(shape[:2], dtype=np.uint8)  # binary board
    for component in components:
        # up and bottom: (column_index, min/max row border)
        for point in component.boundary[0] + component.boundary[1]:
            board[point[1], point[0]] = 255
        # left, right: (row_index, min/max column border)
        for point in component.boundary[2] + component.boundary[3]:
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
