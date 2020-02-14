import numpy as np
import cv2
from collections import Counter

import lib_ip.ip_draw as draw


# detect object(connected region)
def boundary_bfs_connected_area(img, x, y, mark):
    def neighbor(img, x, y, mark, stack):
        for i in range(x - 1, x + 2):
            if i < 0 or i >= img.shape[0]: continue
            for j in range(y - 1, y + 2):
                if j < 0 or j >= img.shape[1]: continue
                if img[i, j] == 255 and mark[i, j] == 0:
                    stack.append([i, j])
                    mark[i, j] = 255

    stack = [[x, y]]  # points waiting for inspection
    area = [[x, y]]  # points of this area
    mark[x, y] = 255  # drawing broad

    while len(stack) > 0:
        point = stack.pop()
        area.append(point)
        neighbor(img, point[0], point[1], mark, stack)
    return area


# get the bounding boundary of an object(region)
# @boundary: [top, bottom, left, right]
# -> up, bottom: (column_index, min/max row border)
# -> left, right: (row_index, min/max column border) detect range of each row
def boundary_get_boundary(area):
    border_up, border_bottom, border_left, border_right = {}, {}, {}, {}
    for point in area:
        # point: (row_index, column_index)
        # up, bottom: (column_index, min/max row border) detect range of each column
        if point[1] not in border_up or border_up[point[1]] > point[0]:
            border_up[point[1]] = point[0]
        if point[1] not in border_bottom or border_bottom[point[1]] < point[0]:
            border_bottom[point[1]] = point[0]
        # left, right: (row_index, min/max column border) detect range of each row
        if point[0] not in border_left or border_left[point[0]] > point[1]:
            border_left[point[0]] = point[1]
        if point[0] not in border_right or border_right[point[0]] < point[1]:
            border_right[point[0]] = point[1]

    boundary = [border_up, border_bottom, border_left, border_right]
    # descending sort
    for i in range(len(boundary)):
        boundary[i] = [[k, boundary[i][k]] for k in boundary[i].keys()]
        boundary[i] = sorted(boundary[i], key=lambda x: x[0])
    return boundary


def boundary_is_line(boundary, min_line_thickness):
    """
    If this object is line by checking its boundary
    :param boundary: boundary: [border_top, border_bottom, border_left, border_right]
                                -> top, bottom: list of (column_index, min/max row border)
                                -> left, right: list of (row_index, min/max column border) detect range of each row
    :param min_line_thickness:
    :return: Boolean
    """
    # horizontally
    slim = 0
    for i in range(len(boundary[0])):
        if abs(boundary[1][i][1] - boundary[0][i][1]) <= min_line_thickness:
            slim += 1
    if slim / len(boundary[0]) > 0.8:
        return True
    # vertically
    slim = 0
    for i in range(len(boundary[2])):
        if abs(boundary[2][i][1] - boundary[3][i][1]) <= min_line_thickness:
            slim += 1
    if slim / len(boundary[2]) > 0.8:
        return True

    return False


# i. detect if an object is rectangle by evenness of each border
# ii. add dent detection
# @boundary: [border_up, border_bottom, border_left, border_right]
# -> up, bottom: (column_index, min/max row border)
# -> left, right: (row_index, min/max column border) detect range of each row
def boundary_is_rectangle(boundary, min_rec_evenness, max_dent_ratio, org_shape=None, show=False):
    dent_direction = [1, -1, 1, -1]  # direction for convex

    flat = 0
    parameter = 0
    for n, border in enumerate(boundary):
        parameter += len(border)
        # dent detection
        pit = 0  # length of pit
        depth = 0  # the degree of surface changing
        if n <= 1:
            adj_side = max(len(boundary[2]), len(boundary[3]))  # get maximum length of adjacent side
        else:
            adj_side = max(len(boundary[0]), len(boundary[1]))

        # -> up, bottom: (column_index, min/max row border)
        # -> left, right: (row_index, min/max column border) detect range of each row
        abnm = 0
        for i in range(3, len(border) - 1):
            # calculate gradient
            difference = border[i][1] - border[i + 1][1]
            # the degree of surface changing
            depth += difference
            # ignore noise at the start of each direction
            if i / len(border) < 0.08 and (dent_direction[n] * difference) / adj_side > 0.5:
                depth = 0  # reset

            # print(border[i][1], i / len(border), depth, (dent_direction[n] * difference) / adj_side )
            # if the change of the surface is too large, count it as part of abnormal change
            if abs(depth) / adj_side > 0.3:
                abnm += 1    # count the size of the abnm
                # if the abnm is too big, the shape should not be a rectangle
                if abnm / len(border) > 0.1:
                    return False
                continue
            else:
                # reset the abnm if the depth back to normal
                abnm = 0

            # if sunken and the surface changing is large, then counted as pit
            if dent_direction[n] * depth < 0 and abs(depth) / adj_side > 0.15:
                pit += 1
                continue

            # if the surface is not changing to a pit and the gradient is zero, then count it as flat
            if abs(depth) < 7:
                flat += 1
            # print(depth, adj_side, abnm)
        # if the pit is too big, the shape should not be a rectangle
        if pit / len(border) > max_dent_ratio:
            return False
        # print()
    # print(flat / parameter, '\n')
    # draw.draw_boundary([boundary], org_shape, show=True)
    # ignore text and irregular shape
    if (flat / parameter) < min_rec_evenness:
        return False
    return True


# @corners: [(top_left, bottom_right)]
# -> top_left: (column_min, row_min)
# -> bottom_right: (column_max, row_max)
def corner_relation(corner_a, corner_b):
    """
    :return: -1 : a in b
             0  : a, b are not intersected
             1  : b in a
             2  : a, b are identical or intersected
    """
    (up_left_a, bottom_right_a) = corner_a
    (y_min_a, x_min_a) = up_left_a
    (y_max_a, x_max_a) = bottom_right_a
    (up_left_b, bottom_right_b) = corner_b
    (y_min_b, x_min_b) = up_left_b
    (y_max_b, x_max_b) = bottom_right_b

    # if a is in b
    if y_min_a > y_min_b and x_min_a > x_min_b and y_max_a < y_max_b and x_max_a < x_max_b:
        return -1
    # if b is in a
    elif y_min_a < y_min_b and x_min_a < x_min_b and y_max_a > y_max_b and x_max_a > x_max_b:
        return 1
    # a and b are non-intersect
    elif (y_min_a > y_max_b or x_min_a > x_max_b) or (y_min_b > y_max_a or x_min_b > x_max_a):
        return 0
    # intersection
    else:
        return 2


def corner_relation_nms(corner_a, corner_b, min_selected_IoU):
    '''
    Calculate the relation between two rectangles by nms
    IoU = Intersection / Union
          0  : Not intersected
          0~1: Overlapped
          1  : Identical
    :return:-2 : b in a and IoU above the threshold
            -1 : a in b
             0 : a, b are not intersected
             1 : b in a
             2 : a in b and IoU above the threshold
             3 : intersected but no containing relation
    '''
    ((col_min_a, row_min_a), (col_max_a, row_max_a)) = corner_a
    ((col_min_b, row_min_b), (col_max_b, row_max_b)) = corner_b

    # get the intersected area
    col_min_s = max(col_min_a, col_min_b)
    row_min_s = max(row_min_a, row_min_b)
    col_max_s = min(col_max_a, col_max_b)
    row_max_s = min(row_max_a, row_max_b)
    w = np.maximum(0, col_max_s - col_min_s)
    h = np.maximum(0, row_max_s - row_min_s)
    inter = w * h
    area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
    area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
    iou = inter / (area_a + area_b - inter)

    # not intersected with each other
    if iou == 0:
        return 0
    # overlapped too much with each other
    if iou > 0.6:
        # a in b
        if inter == area_a:
            return -2
        # b in a
        if inter == area_b:
            return 2
    # intersected and containing relation
    if min_selected_IoU < iou <= 0.6:
        # a in b
        if inter == area_a:
            return -1
        # b in a
        if inter == area_b:
            return 1
    # containing but too small
    if iou <= min_selected_IoU:
        # a in b
        if inter == area_a:
            return -3
        # b in a
        if inter == area_b:
            return 3

    # intersected but no containing relation
    return 4


def corner_cvt_relative_position(corners, col_min_base, row_min_base):
    """
    get the relative position of corners in the entire image
    """
    rlt_corners = []
    for corner in corners:
        (top_left, bottom_right) = corner
        (col_min, row_min) = top_left
        (col_max, row_max) = bottom_right
        col_min += col_min_base
        col_max += col_min_base
        row_min += row_min_base
        row_max += row_min_base
        rlt_corners.append(((col_min, row_min), (col_max, row_max)))

    return rlt_corners


def corner_merge_two_corners(corner_a, corner_b):
    ((col_min_a, row_min_a), (col_max_a, row_max_a)) = corner_a
    ((col_min_b, row_min_b), (col_max_b, row_max_b)) = corner_b

    col_min = min(col_min_a, col_min_b)
    col_max = max(col_max_a, col_max_b)
    row_min = min(row_min_a, row_min_b)
    row_max = max(row_max_a, row_max_b)
    return (col_min, row_min), (col_max, row_max)


def line_check_perpendicular(lines_h, lines_v, max_thickness):
    """
    lines: [line_h, line_v]
        -> line_h: horizontal {'head':(column_min, row), 'end':(column_max, row), 'thickness':int)
        -> line_v: vertical {'head':(column, row_min), 'end':(column, row_max), 'thickness':int}
    """
    is_per_h = np.full(len(lines_h), False)
    is_per_v = np.full(len(lines_v), False)
    for i in range(len(lines_h)):
        # save the intersection point of h
        lines_h[i]['inter_point'] = set()
        h = lines_h[i]

        for j in range(len(lines_v)):
            # save the intersection point of v
            if 'inter_point' not in lines_v[j]: lines_v[j]['inter_point'] = set()
            v = lines_v[j]

            # if h is perpendicular to v in head of v
            if abs(h['head'][1]-v['head'][1]) <= max_thickness:
                if abs(h['head'][0] - v['head'][0]) <= max_thickness:
                    lines_h[i]['inter_point'].add('head')
                    lines_v[j]['inter_point'].add('head')
                    is_per_h[i] = True
                    is_per_v[j] = True
                elif abs(h['end'][0] - v['head'][0]) <= max_thickness:
                    lines_h[i]['inter_point'].add('end')
                    lines_v[j]['inter_point'].add('head')
                    is_per_h[i] = True
                    is_per_v[j] = True

            # if h is perpendicular to v in end of v
            elif abs(h['head'][1]-v['end'][1]) <= max_thickness:
                if abs(h['head'][0] - v['head'][0]) <= max_thickness:
                    lines_h[i]['inter_point'].add('head')
                    lines_v[j]['inter_point'].add('end')
                    is_per_h[i] = True
                    is_per_v[j] = True
                elif abs(h['end'][0] - v['head'][0]) <= max_thickness:
                    lines_h[i]['inter_point'].add('end')
                    lines_v[j]['inter_point'].add('end')
                    is_per_h[i] = True
                    is_per_v[j] = True
    per_h = []
    per_v = []
    for i in range(len(is_per_h)):
        if is_per_h[i]:
            lines_h[i]['inter_point'] = list(lines_h[i]['inter_point'])
            per_h.append(lines_h[i])
    for i in range(len(is_per_v)):
        if is_per_v[i]:
            lines_v[i]['inter_point'] = list(lines_v[i]['inter_point'])
            per_v.append(lines_v[i])
    return per_h, per_v


def line_shrink_corners(corner, lines_h, lines_v):
    """
    shrink the corner according to lines:
             col_min_shrink: shrink right (increase)
             col_max_shrink: shrink left  (decrease)
             row_min_shrink: shrink down  (increase)
             row_max_shrink: shrink up    (decrease)
    :param lines_h: horizontal {'head':(column_min, row), 'end':(column_max, row), 'thickness':int)
    :param lines_v: vertical {'head':(column, row_min), 'end':(column, row_max), 'thickness':int}
    :return: shrunken corner: (top_left, bottom_right)
    """
    (col_min, row_min), (col_max, row_max) = corner
    col_min_shrink, row_min_shrink = col_min, row_min
    col_max_shrink, row_max_shrink = col_max, row_max
    valid_frame = False

    for h in lines_h:
        # ignore outer border
        if len(h['inter_point']) == 2:
            valid_frame = True
            continue
        # shrink right -> col_min move to end
        if h['inter_point'][0] == 'head':
            col_min_shrink = max(h['end'][0], col_min_shrink)
        # shrink left -> col_max move to head
        elif h['inter_point'][0] == 'end':
            col_max_shrink = min(h['head'][0], col_max_shrink)

    for v in lines_v:
        # ignore outer border
        if len(v['inter_point']) == 2:
            valid_frame = True
            continue
        # shrink down -> row_min move to end
        if v['inter_point'][0] == 'head':
            row_min_shrink = max(v['end'][1], row_min_shrink)
        # shrink up -> row_max move to head
        elif v['inter_point'][0] == 'end':
            row_max_shrink = min(v['head'][1], row_max_shrink)

    # return the shrunken corner if only there is line intersecting with two other lines
    if valid_frame:
        return (col_min_shrink, row_min_shrink), (col_max_shrink, row_max_shrink)
    return corner


def line_cvt_relative_position(col_min, row_min, lines_h, lines_v):
    """
    convert the relative position of lines in the entire image
    :param col_min: based column the img lines belong to
    :param row_min: based row the img lines belong to
    :param lines_h: horizontal {'head':(column_min, row), 'end':(column_max, row), 'thickness':int)
    :param lines_v: vertical {'head':(column, row_min), 'end':(column, row_max), 'thickness':int}
    :return: lines_h_cvt, lines_v_cvt
    """
    for h in lines_h:
        h['head'][0] += col_min
        h['head'][1] += row_min
        h['end'][0] += col_min
        h['end'][1] += row_min
    for v in lines_v:
        v['head'][0] += col_min
        v['head'][1] += row_min
        v['end'][0] += col_min
        v['end'][1] += row_min

    return lines_h, lines_v


# check if an object is so slim
# @boundary: [border_up, border_bottom, border_left, border_right]
# -> up, bottom: (column_index, min/max row border)
# -> left, right: (row_index, min/max column border) detect range of each row
def clipping_by_line(boundary, boundary_rec, lines):
    boundary = boundary.copy()
    for orient in lines:
        # horizontal
        if orient == 'h':
            # column range of sub area
            r1, r2 = 0, 0
            for line in lines[orient]:
                if line[0] == 0:
                    r1 = line[1]
                    continue
                r2 = line[0]
                b_top = []
                b_bottom = []
                for i in range(len(boundary[0])):
                    if r2 > boundary[0][i][0] >= r1:
                        b_top.append(boundary[0][i])
                for i in range(len(boundary[1])):
                    if r2 > boundary[1][i][0] >= r1:
                        b_bottom.append(boundary[1][i])

                b_left = [x for x in boundary[2]]  # (row_index, min column border)
                for i in range(len(b_left)):
                    if b_left[i][1] < r1:
                        b_left[i][1] = r1
                b_right = [x for x in boundary[3]]  # (row_index, max column border)
                for i in range(len(b_right)):
                    if b_right[i][1] > r2:
                        b_right[i][1] = r2

                boundary_rec.append([b_top, b_bottom, b_left, b_right])
                r1 = line[1]
