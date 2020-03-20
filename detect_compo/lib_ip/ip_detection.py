import cv2
import numpy as np

import lib_ip.ip_draw as draw
import lib_ip.ip_preprocessing as pre
import lib_ip.ip_detection_utils as util
import lib_ip.ocr_classify_text as ocr
from lib_ip.Component import Component
import lib_ip.Component as Compo
from config.CONFIG_UIED import Config
C = Config()


def merge_intersected_corner(compos, org_shape, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    changed = False
    new_compos= []
    row, col = org_shape[:2]
    for i in range(len(compos)):
        merged = False
        # if compos[i].height / row > max_compo_scale[0]:
        #     new_compos.append(compos[i])
        #     continue
        for j in range(len(new_compos)):
            # if compos[j].height / row > max_compo_scale[0]:
            #     continue
            relation = compos[i].compo_relation(compos[j])
            if relation == 2:
                new_compos[j].compo_merge(compos[i])
                merged = True
                changed = True
                break
        if not merged:
            new_compos.append(compos[i])

    Compo.compos_update(compos)
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
                 show=True):
    width = binary.shape[1]
    thickness = 0
    gap = 0
    broad = np.zeros(binary.shape[:2], dtype=np.uint8)

    line_length = 0
    start, end = -1, -1
    for i, row in enumerate(binary):
        # line_cut = 0
        # for j, point in enumerate(row):
        #     if point != 0:
        #         line_cut = 0
        #         line_length += 1
        #     else:
        #         line_cut += 1
        #         if line_cut >= 5:
        #             if j > width * (1 - min_line_length_ratio):
        #                 break

        if (sum(row) / 255) / width > min_line_length_ratio:
            # print(gap, start, end)
            # broad[i] = binary[i]
            # cv2.imshow('line', broad)
            # cv2.waitKey()
            gap = 0
            if start == -1:
                start = i
            line_length = max(line_length, sum(row) / 255)
        else:
            if (sum(row) / 255) / width < 0.3:
                gap += 1
                if start != -1 and end == -1:
                    end = i
                if 0 < end - start < max_line_thickness and gap >= 1:
                    # print(line_length / width, end - start, (sum(row) / 255) / width,
                    #       (sum(binary[i - thickness]) / 255) / width)
                    # broad[start: end] = binary[start: end]
                    # cv2.imshow('line', broad)
                    # cv2.waitKey()
                    binary[start: end] = 0
                    start, end = -1, -1
            else:
                if 0 < end - start < max_line_thickness and gap >= 1:
                    # print(line_length / width, end - start, (sum(row) / 255) / width,
                    #       (sum(binary[i - thickness]) / 255) / width)
                    # broad[start: end] = binary[start: end]
                    # cv2.imshow('line', broad)
                    # cv2.waitKey()
                    binary[start: end] = 0
                start, end = -1, -1
    if show:
        cv2.imshow('no-line', binary)
        cv2.waitKey()


def rm_noise_compos(compos):
    compos_new = []
    for compo in compos:
        if compo.category == 'Noise':
            continue
        compos_new.append(compo)
    return compos_new


def rm_noise_in_large_img(compos, org,
                      max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    row, column = org.shape[:2]
    remain = np.full(len(compos), True)
    new_compos = []
    for compo in compos:
        if compo.category == 'Image':
            for i in compo.contain:
                remain[i] = False
    for i in range(len(remain)):
        if remain[i]:
            new_compos.append(compos[i])
    return new_compos


def detect_compos_in_img(compos, binary, org, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE, show=False):
    compos_new = []
    row, column = binary.shape[:2]
    for compo in compos:
        if compo.category == 'Image':
            compo.compo_update_bbox_area()
            # org_clip = compo.compo_clipping(org)
            # bin_clip = pre.binarization(org_clip, show=show)
            bin_clip = compo.compo_clipping(binary)
            bin_clip = pre.reverse_binary(bin_clip, show=show)

            compos_rec, compos_nonrec = component_detection(bin_clip, test=False, step_h=10, step_v=10, rec_detect=True)
            for compo_rec in compos_rec:
                compo_rec.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
                if compo_rec.bbox_area / compo.bbox_area < 0.8 and compo_rec.bbox.height > 20 and compo_rec.bbox.width > 20:
                    compos_new.append(compo_rec)
                    # draw.draw_bounding_box(org, [compo_rec], show=True)

            # compos_inner = component_detection(bin_clip, rec_detect=False)
            # for compo_inner in compos_inner:
            #     compo_inner.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
            #     draw.draw_bounding_box(org, [compo_inner], show=True)
            #     if compo_inner.bbox_area / compo.bbox_area < 0.8:
            #         compos_new.append(compo_inner)
    compos += compos_new


def compo_filter(compos, org):
    compos_new = []
    for compo in compos:
        if compo.height < 26 and compo.width < 26:
            continue
        if compo.category == 'TextView' and compo.height > 100 and compo.width / org.shape[1] < 0.9:
            compo.category = 'ImageView'
        compos_new.append(compo)
    return compos_new


# take the binary image as input
# calculate the connected regions -> get the bounding boundaries of them -> check if those regions are rectangles
# return all boundaries and boundaries of rectangles
def component_detection(binary,
                        min_obj_area=C.THRESHOLD_OBJ_MIN_AREA,
                        line_thickness=C.THRESHOLD_LINE_THICKNESS,
                        min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                        max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                        step_h = 5, step_v = 2,
                        rec_detect=False, show=False, test=False):
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
    for i in range(0, row, step_h):
        for j in range(i % 2, column, step_v):
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
                if test:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary([component], binary.shape, show=True)
                # check if it is line by checking the length of edges
                if component.area > min_obj_area * 5 and component.compo_is_line(line_thickness):
                    continue
                compos_all.append(component)

                if rec_detect:
                    # rectangle check
                    if component.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                        component.rect_ = True
                        compos_rec.append(component)
                    else:
                        component.rect_ = False
                        compos_nonrec.append(component)

                if show:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary(compos_all, binary.shape, show=True)

    # draw.draw_boundary(compos_all, binary.shape, show=True)
    if rec_detect:
        return compos_rec, compos_nonrec
    else:
        return compos_all
