import cv2
import numpy as np

import lib_ip.ip_draw as draw
import lib_ip.ip_preprocessing as pre
import lib_ip.ip_detection_utils as util
import lib_ip.ocr_classify_text as ocr
from config.CONFIG_UIED import Config
C = Config()


def corner_padding(img, corners, pad):
    row, col = img.shape[:2]
    corners_new = []
    for corner in corners:
        ((column_min, row_min), (column_max, row_max)) = corner
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, col)
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, row)
        corners_new.append(((column_min, row_min), (column_max, row_max)))
    return corners_new


def get_corner(boundaries):
    """
    Get the top left and bottom right points of boundary
    :param boundaries: boundary: [top, bottom, left, right]
                        -> up, bottom: (column_index, min/max row border)
                        -> left, right: (row_index, min/max column border) detect range of each row
    :return: corners: [(top_left, bottom_right)]
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    """
    corners = []
    for boundary in boundaries:
        top_left = (int(min(boundary[0][0][0], boundary[1][-1][0])), int(min(boundary[2][0][0], boundary[3][-1][0])))
        bottom_right = (int(max(boundary[0][0][0], boundary[1][-1][0])), int(max(boundary[2][0][0], boundary[3][-1][0])))
        corner = (top_left, bottom_right)
        corners.append(corner)
    return corners


def select_corner(corners, compos_class, class_name):
    """
    Select corners in given compo type
    """
    corners_wanted = []
    for i in range(len(compos_class)):
        if compos_class[i] == class_name:
            corners_wanted.append(corners[i])
    return corners_wanted


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


def merge_intersected_corner(corners, org_shape, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    def is_intersected(corner_a, corner_b):
        ((col_min_a, row_min_a), (col_max_a, row_max_a)) = corner_a
        ((col_min_b, row_min_b), (col_max_b, row_max_b)) = corner_b
        area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
        area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)

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
    new_corners = []
    row, col = org_shape[:2]
    for i in range(len(corners)):
        merged = False
        height = corners[i][1][1] - corners[i][0][1]
        if height / row > max_compo_scale[0]:
            new_corners.append(corners[i])
            continue
        for j in range(len(new_corners)):
            if (corners[j][1][1] - corners[j][0][1]) / row > max_compo_scale[0]:
                continue
            if is_intersected(corners[i], new_corners[j]):
                new_corners[j] = util.corner_merge_two_corners(corners[i], new_corners[j])
                merged = True
                changed = True
                break
        if not merged:
            new_corners.append(corners[i])

    if not changed:
        return corners
    else:
        return merge_intersected_corner(new_corners, org_shape)


def merge_text(corners, org_shape, max_word_gad=C.THRESHOLD_TEXT_MAX_WORD_GAP, max_word_height_ratio=C.THRESHOLD_TEXT_MAX_HEIGHT):
    def is_text_line(corner_a, corner_b):
        ((col_min_a, row_min_a), (col_max_a, row_max_a)) = corner_a
        ((col_min_b, row_min_b), (col_max_b, row_max_b)) = corner_b
        # on the same line
        if abs(row_min_a - row_min_b) < max_word_gad and abs(row_max_a - row_max_b) < max_word_gad:
            # close distance
            if abs(col_min_b - col_max_a) < max_word_gad or abs(col_min_a - col_max_b) < max_word_gad:
                return True
        return False

    changed = False
    new_corners = []
    row, col = org_shape[:2]
    for i in range(len(corners)):
        merged = False
        height = corners[i][1][1] - corners[i][0][1]
        # ignore non-text
        if height / row > max_word_height_ratio:
            new_corners.append(corners[i])
            continue
        for j in range(len(new_corners)):
            if is_text_line(corners[i], new_corners[j]):
                new_corners[j] = util.corner_merge_two_corners(corners[i], new_corners[j])
                merged = True
                changed = True
                break
        if not merged:
            new_corners.append(corners[i])

    if not changed:
        return corners
    else:
        return merge_text(new_corners, org_shape)


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


def is_top_or_bottom_bar(corner, org, top_bottom_height=C.THRESHOLD_TOP_BOTTOM_BAR):
    height, width = org.shape[:2]
    ((column_min, row_min), (column_max, row_max)) = corner
    if column_min < 5 and row_min < 5 and \
            width - column_max < 5 and row_max < height * top_bottom_height[0]:
        return True
    if column_min < 5 and row_min > height * top_bottom_height[1] and \
            width - column_max < 5 and height - row_max < 5:
        return True
    return False


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


def rm_top_or_bottom_corners(corners, org_shape, top_bottom_height=C.THRESHOLD_TOP_BOTTOM_BAR):
    new_corners = []
    height, width = org_shape[:2]
    for corner in corners:
        ((column_min, row_min), (column_max, row_max)) = corner
        # remove big ones
        # if (row_max - row_min) / height > 0.65 and (column_max - column_min) / width > 0.8:
        #     continue
        if not (row_max < height * top_bottom_height[0] or row_min > height * top_bottom_height[1]):
            new_corners.append(corner)
    return new_corners


def line_removal(binary,
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
                if line_cut >= 5:
                    if j > width * (1 - min_line_length_ratio):
                        break

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
def boundary_detection(binary,
                       min_obj_area=C.THRESHOLD_OBJ_MIN_AREA, min_obj_perimeter=C.THRESHOLD_OBJ_MIN_PERIMETER,
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
    boundary_all = []
    boundary_rec = []
    boundary_nonrec = []
    row, column = binary.shape[0], binary.shape[1]

    for i in range(0, row, 5):
        for j in range(i%2, column, 2):
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
                # calculate the boundary of the connected area
                boundary = util.boundary_get_boundary(region)
                # ignore small area
                if len(boundary[0]) <= 3 or len(boundary[2]) <= 3:
                    continue
                # print('Area:%d' % (len(region)))
                # draw.draw_boundary([boundary], binary.shape, show=False)
                # check if it is line by checking the length of edges
                if len(region) > min_obj_area * 10 and util.boundary_is_line(boundary, line_thickness):
                    continue
                boundary_all.append(boundary)

                if rec_detect:
                    # rectangle check
                    if util.boundary_is_rectangle(boundary, min_rec_evenness, max_dent_ratio):
                        boundary_rec.append(boundary)
                    else:
                        boundary_nonrec.append(boundary)

                if show:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary(boundary_all, binary.shape, show=True)

    # draw.draw_boundary(boundary_all, binary.shape, show=True)
    if rec_detect:
        return boundary_rec, boundary_nonrec
    else:
        return boundary_all
