import numpy as np
import cv2
from collections import Counter

import lib_ip.ip_draw as draw
from config.CONFIG_UIED import Config
C = Config()


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


# remove imgs that contain text
def rm_text(org, corners, compo_class,
            max_text_height=C.THRESHOLD_TEXT_MAX_HEIGHT, max_text_width=C.THRESHOLD_TEXT_MAX_WIDTH,
            ocr_padding=C.OCR_PADDING, ocr_min_word_area=C.OCR_MIN_WORD_AREA, show=False):
    """
    Remove area that full of text
    :param org: original image
    :param corners: [(top_left, bottom_right)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param compo_class: classes of corners
    :param max_text_height: Too large to be text
    :param max_text_width: Too large to be text
    :param ocr_padding: Padding for clipping
    :param ocr_min_word_area: If too text area ratio is too large
    :param show: Show or not
    :return: corners without text objects
    """
    new_corners = []
    new_class = []
    for i in range(len(corners)):
        corner = corners[i]
        (top_left, bottom_right) = corner
        (col_min, row_min) = top_left
        (col_max, row_max) = bottom_right
        height = row_max - row_min
        width = col_max - col_min
        # highly likely to be block or img if too large
        if height > max_text_height and width > max_text_width:
            new_corners.append(corner)
            new_class.append(compo_class[i])
        else:
            row_min = row_min - ocr_padding if row_min - ocr_padding >= 0 else 0
            row_max = row_max + ocr_padding if row_max + ocr_padding < org.shape[0] else org.shape[0]
            col_min = col_min - ocr_padding if col_min - ocr_padding >= 0 else 0
            col_max = col_max + ocr_padding if col_max + ocr_padding < org.shape[1] else org.shape[1]
            # check if this area is text
            clip = org[row_min: row_max, col_min: col_max]
            if not ocr.is_text(clip, ocr_min_word_area, show=show):
                new_corners.append(corner)
                new_class.append(compo_class[i])
    return new_corners, new_class


def rm_img_in_compo(corners_img, corners_compo):
    """
    Remove imgs in component
    """
    corners_img_new = []
    for img in corners_img:
        is_nested = False
        for compo in corners_compo:
            if util.corner_relation(img, compo) == -1:
                is_nested = True
                break
        if not is_nested:
            corners_img_new.append(img)
    return corners_img_new


def block_or_compo(org, binary, corners,
                   max_thickness=C.THRESHOLD_BLOCK_MAX_BORDER_THICKNESS, max_block_cross_points=C.THRESHOLD_BLOCK_MAX_CROSS_POINT,
                   min_compo_w_h_ratio=C.THRESHOLD_UICOMPO_MIN_W_H_RATIO, max_compo_w_h_ratio=C.THRESHOLD_UICOMPO_MAX_W_H_RATIO,
                   min_block_edge=C.THRESHOLD_BLOCK_MIN_EDGE_LENGTH):
    """
    Check if the objects are img components or just block
    :param org: Original image
    :param binary:  Binary image from pre-processing
    :param corners: [(top_left, bottom_right)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param max_thickness: The max thickness of border of blocks
    :param max_block_cross_points: Ratio of point of interaction
    :return: corners of blocks and imgs
    """
    blocks = []
    imgs = []
    compos = []
    for corner in corners:
        (top_left, bottom_right) = corner
        (col_min, row_min) = top_left
        (col_max, row_max) = bottom_right
        height = row_max - row_min
        width = col_max - col_min

        block = False
        vacancy = [0, 0, 0, 0]
        for i in range(1, max_thickness):
            try:
                # top to bottom
                if vacancy[0] == 0 and (col_max - col_min - 2 * i) is not 0 and (
                        np.sum(binary[row_min + i, col_min + i: col_max - i]) / 255) / (col_max - col_min - 2 * i) <= max_block_cross_points:
                    vacancy[0] = 1
                # bottom to top
                if vacancy[1] == 0 and (col_max - col_min - 2 * i) is not 0 and (
                        np.sum(binary[row_max - i, col_min + i: col_max - i]) / 255) / (col_max - col_min - 2 * i) <= max_block_cross_points:
                    vacancy[1] = 1
                # left to right
                if vacancy[2] == 0 and (row_max - row_min - 2 * i) is not 0 and (
                        np.sum(binary[row_min + i: row_max - i, col_min + i]) / 255) / (row_max - row_min - 2 * i) <= max_block_cross_points:
                    vacancy[2] = 1
                # right to left
                if vacancy[3] == 0 and (row_max - row_min - 2 * i) is not 0 and (
                        np.sum(binary[row_min + i: row_max - i, col_max - i]) / 255) / (row_max - row_min - 2 * i) <= max_block_cross_points:
                    vacancy[3] = 1
                if np.sum(vacancy) == 4:
                    block = True
            except:
                pass

        # too big to be UI components
        if block:
            if height > min_block_edge and width > min_block_edge:
                blocks.append(corner)
            else:
                if min_compo_w_h_ratio < width / height < max_compo_w_h_ratio:
                    compos.append(corner)
        # filter out small objects
        else:
            if height > min_block_edge:
                imgs.append(corner)
            else:
                if min_compo_w_h_ratio < width / height < max_compo_w_h_ratio:
                    compos.append(corner)
    return blocks, imgs, compos


def compo_on_img(processing, org, binary, clf,
                 compos_corner, compos_class):
    """
    Detect potential UI components inner img;
    Only leave non-img
    """
    pad = 2
    for i in range(len(compos_corner)):
        if compos_class[i] != 'img':
            continue
        ((col_min, row_min), (col_max, row_max)) = compos_corner[i]
        col_min = max(col_min - pad, 0)
        col_max = min(col_max + pad, org.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, org.shape[0])
        area = (col_max - col_min) * (row_max - row_min)
        if area < 600:
            continue

        clip_org = org[row_min:row_max, col_min:col_max]
        clip_bin_inv = pre.reverse_binary(binary[row_min:row_max, col_min:col_max])

        compos_boundary_new, compos_corner_new, compos_class_new = processing(clip_org, clip_bin_inv, clf)
        compos_corner_new = util.corner_cvt_relative_position(compos_corner_new, col_min, row_min)

        assert len(compos_corner_new) == len(compos_class_new)

        # only leave non-img elements
        for i in range(len(compos_corner_new)):
            ((col_min_new, row_min_new), (col_max_new, row_max_new)) = compos_corner_new[i]
            area_new = (col_max_new - col_min_new) * (row_max_new - row_min_new)
            if compos_class_new[i] != 'img' and area_new / area < 0.8:
                compos_corner.append(compos_corner_new[i])
                compos_class.append(compos_class_new[i])

    return compos_corner, compos_class


def strip_img(corners_compo, compos_class, corners_img):
    """
    Separate img from other compos
    :return: compos without img
    """
    corners_compo_withuot_img = []
    compo_class_withuot_img = []
    for i in range(len(compos_class)):
        if compos_class[i] == 'img':
            corners_img.append(corners_compo[i])
        else:
            corners_compo_withuot_img.append(corners_compo[i])
            compo_class_withuot_img.append(compos_class[i])
    return corners_compo_withuot_img, compo_class_withuot_img


def merge_corner(corners, compos_class, min_selected_IoU=C.THRESHOLD_MIN_IOU, is_merge_nested_same=True):
    """
    Calculate the Intersection over Overlap (IoU) and merge corners according to the value of IoU
    :param is_merge_nested_same: if true, merge the nested corners with same class whatever the IoU is
    :param corners: corners: [(top_left, bottom_right)]
                            -> top_left: (column_min, row_min)
                            -> bottom_right: (column_max, row_max)
    :return: new corners
    """
    new_corners = []
    new_class = []
    for i in range(len(corners)):
        is_intersected = False
        for j in range(len(new_corners)):
            r = util.corner_relation_nms(corners[i], new_corners[j], min_selected_IoU)
            # r = util.corner_relation(corners[i], new_corners[j])
            if is_merge_nested_same:
                if compos_class[i] == new_class[j]:
                    # if corners[i] is in new_corners[j], ignore corners[i]
                    if r == -1:
                        is_intersected = True
                        break
                    # if new_corners[j] is in corners[i], replace new_corners[j] with corners[i]
                    elif r == 1:
                        is_intersected = True
                        new_corners[j] = corners[i]

            # if above IoU threshold, and corners[i] is in new_corners[j], ignore corners[i]
            if r == -2:
                is_intersected = True
                break
            # if above IoU threshold, and new_corners[j] is in corners[i], replace new_corners[j] with corners[i]
            elif r == 2:
                is_intersected = True
                new_corners[j] = corners[i]
                new_class[j] = compos_class[i]

            # containing and too small
            elif r == -3:
                is_intersected = True
                break
            elif r == 3:
                is_intersected = True
                new_corners[j] = corners[i]

            # if [i] and [j] are overlapped but no containing relation, merge corners when same class
            elif r == 4:
                is_intersected = True
                if compos_class[i] == new_class[j]:
                    new_corners[j] = util.corner_merge_two_corners(corners[i], new_corners[j])

        if not is_intersected:
            new_corners.append(corners[i])
            new_class.append(compos_class[i])
    return new_corners, new_class


def select_corner(corners, compos_class, class_name):
    """
    Select corners in given compo type
    """
    corners_wanted = []
    for i in range(len(compos_class)):
        if compos_class[i] == class_name:
            corners_wanted.append(corners[i])
    return corners_wanted


def flood_fill_bfs(img, x_start, y_start, mark, grad_thresh):
    def neighbor(x, y):
        for i in range(x - 1, x + 2):
            if i < 0 or i >= img.shape[0]: continue
            for j in range(y - 1, y + 2):
                if j < 0 or j >= img.shape[1]: continue
                if mark[i, j] == 0 and abs(img[i, j] - img[x, y]) < grad_thresh:
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