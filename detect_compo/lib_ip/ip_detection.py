import cv2
import numpy as np

import lib_ip.ip_draw as draw
import lib_ip.ip_preprocessing as pre
import lib_ip.ip_detection_utils as util
import lib_ip.ocr_classify_text as ocr
from lib_ip.Component import Component
from config.CONFIG_UIED import Config
C = Config()


def merge_intersected_corner(compos, org_shape, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    def is_intersected(compo_a, compo_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = compo_a.put_bbox()
        (col_min_b, row_min_b, col_max_b, row_max_b) = compo_b.put_bbox()
        area_a = compo_a.area
        area_b = compo_b.area

        # get the intersected area
        col_min_s = max(col_min_a, col_min_b)
        row_min_s = max(row_min_a, row_min_b)
        col_max_s = min(col_max_a, col_max_b)
        row_max_s = min(row_max_a, row_max_b)
        w = max(0, col_max_s - col_min_s)
        h = max(0, row_max_s - row_min_s)
        inter = w * h
        # intersected but not containing
        if inter == 0 and inter != area_a and inter != area_b:
            return False
        # very closed
        return True

    changed = False
    new_compos= []
    row, col = org_shape[:2]
    for i in range(len(compos)):
        merged = False
        height = compos[i].height
        if height / row > max_compo_scale[0]:
            new_compos.append(compos[i])
            continue
        for j in range(len(new_compos)):
            if compos[j].height / row > max_compo_scale[0]:
                continue
            relation = compos[i].compo_relation(compos[j])
            if relation == 2:
                new_compos[j].compo_merge(compos[i])
                merged = True
                changed = True
                break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_intersected_corner(new_compos, org_shape)


def merge_text(compos, org_shape, max_word_gad=C.THRESHOLD_TEXT_MAX_WORD_GAP, max_word_height_ratio=C.THRESHOLD_TEXT_MAX_HEIGHT):
    def is_text_line(compo_a, compo_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = compo_a.put_bbox()
        (col_min_b, row_min_b, col_max_b, row_max_b) = compo_b.put_bbox()
        # on the same line
        if abs(row_min_a - row_min_b) < max_word_gad and abs(row_max_a - row_max_b) < max_word_gad:
            # close distance
            if abs(col_min_b - col_max_a) < max_word_gad or abs(col_min_a - col_max_b) < max_word_gad:
                return True
        return False

    changed = False
    new_compos = []
    row, col = org_shape[:2]
    for i in range(len(compos)):
        merged = False
        height = compos[i].height
        # ignore non-text
        # if height / row > max_word_height_ratio\
        #         or compos[i].category != 'Text':
        if height > 26:
            new_compos.append(compos[i])
            continue
        for j in range(len(new_compos)):
            # if compos[j].category != 'Text':
            #     continue
            if is_text_line(compos[i], new_compos[j]):
                new_compos[j].compo_merge(compos[i])
                merged = True
                changed = True
                break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_text(new_compos, org_shape)


def rm_top_or_bottom_corners(components, org_shape, top_bottom_height=C.THRESHOLD_TOP_BOTTOM_BAR):
    new_compos = []
    height, width = org_shape[:2]
    for compo in components:
        (column_min, row_min, column_max, row_max) = compo.put_bbox()
        # remove big ones
        # if (row_max - row_min) / height > 0.65 and (column_max - column_min) / width > 0.8:
        #     continue
        if not (row_max < height * top_bottom_height[0] or row_min > height * top_bottom_height[1]):
            new_compos.append(compo)
    return new_compos


def rm_line(binary,
                 max_line_thickness=C.THRESHOLD_LINE_THICKNESS,
                 min_line_length_ratio=C.THRESHOLD_LINE_MIN_LENGTH,
                 show=False):
    width = binary.shape[1]
    thickness = 0
    gap = 0
    broad = np.zeros(binary.shape[:2], dtype=np.uint8)

    for i, row in enumerate(binary):
        line_length = 0
        line_cut = 0
        for j, point in enumerate(row):
            broad[i][j] = point
            if point != 0:
                line_cut = 0
                line_length += 1
            else:
                line_cut += 1
                # if line_cut >= 5:
                #     if j > width * (1 - min_line_length_ratio):
                #         break

        if line_length / width > min_line_length_ratio:
            gap = 0
            thickness += 1
        elif (sum(row) / 255) / width < 0.5:
            gap += 1
            if thickness > 0:
                # line ends
                if thickness <= max_line_thickness:
                    # erase line part if line is detected
                    binary[i - thickness: i] = 0
                    thickness = 0
                if gap >= max_line_thickness:
                    thickness = 0
    if show:
        cv2.imshow('no-line', binary)
        cv2.waitKey()


# take the binary image as input
# calculate the connected regions -> get the bounding boundaries of them -> check if those regions are rectangles
# return all boundaries and boundaries of rectangles
def component_detection(binary,
                       min_obj_area=C.THRESHOLD_OBJ_MIN_AREA,
                       line_thickness=C.THRESHOLD_LINE_THICKNESS,
                       min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                       max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                       rec_detect=False, show=False):
    """
    :param binary: Binary image from pre-processing
    :param min_obj_area: If not pass then ignore the small object
    :param min_obj_perimeter: If not pass then ignore the small object
    :param line_thickness: If not pass then ignore the slim object
    :param min_rec_evenness: If not pass then this object cannot be rectangular
    :param max_dent_ratio: If not pass then this object cannot be rectangular
    :return: boundary: [top, bottom, left, right]
                        -> up, bottom: list of (column_index, min/max row border)
                        -> left, right: list of (row_index, min/max column border) detect range of each row
    """
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    compos_all = []
    compos_rec = []
    compos_nonrec = []
    row, column = binary.shape[0], binary.shape[1]
    for i in range(0, row, 5):
        for j in range(i % 2, column, 2):
            if binary[i, j] == 255 and mask[i, j] == 0:
                # get connected area
                # region = util.boundary_bfs_connected_area(binary, i, j, mask)

                mask_copy = mask.copy()
                cv2.floodFill(binary, mask, (j, i), None, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                mask_copy = mask - mask_copy
                region = np.nonzero(mask_copy[1:-1, 1:-1])
                region = list(zip(region[0], region[1]))

                # ignore small area
                if len(region) < min_obj_area:
                    continue
                component = Component(region, binary.shape)
                # calculate the boundary of the connected area
                # ignore small area
                if component.width <= 3 or component.height <= 3:
                    continue
                # print('Area:%d' % (len(region)))
                # draw.draw_boundary([component], binary.shape, show=True)
                # check if it is line by checking the length of edges
                if component.area > min_obj_area * 10 and component.compo_is_line(line_thickness):
                    continue
                compos_all.append(component)

                if rec_detect:
                    # rectangle check
                    if component.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                        compos_rec.append(component)
                    else:
                        compos_nonrec.append(component)

                if show:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary(compos_all, binary.shape, show=True)

    # draw.draw_boundary(boundary_all, binary.shape, show=True)
    if rec_detect:
        return compos_rec, compos_nonrec
    else:
        return compos_all
