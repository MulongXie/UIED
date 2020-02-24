import json
import cv2
import numpy as np
from os.path import join as pjoin
import os

import lib_ip.ip_preprocessing as pre
import lib_ip.file_utils as file
from config.CONFIG import Config
C = Config()


def draw_bounding_box_class(org, corners, compo_class, color_map=C.COLOR, line=2, show=False, name='img'):
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color_map[compo_class[i]], line)
        board = cv2.putText(board, compo_class[i], (corners[i][0]+5, corners[i][1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[compo_class[i]], 2)
    if show:
        cv2.imshow(name, board)
        cv2.waitKey(0)
    return board


def draw_bounding_box(org, corners, color=(0, 255, 0), line=3, show=False):
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color, line)
    if show:
        cv2.imshow('a', board)
        cv2.waitKey(0)
    return board


def save_corners_json(file_path, corners, compo_classes, new=True):
    if not new:
        f_in = open(file_path, 'r')
        components = json.load(f_in)
    else:
        components = {'compos': []}
    f_out = open(file_path, 'w')

    for i in range(len(corners)):
        c = {'class': compo_classes[i], 'column_min': corners[i][0], 'row_min': corners[i][1],
             'column_max': corners[i][2], 'row_max': corners[i][3]}
        components['compos'].append(c)

    json.dump(components, f_out, indent=4)


def nms(org, corners_compo_old, compos_class_old, corner_text):
    def merge_two_corners(corner_a, corner_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = corner_a
        (col_min_b, row_min_b, col_max_b, row_max_b) = corner_b

        col_min = min(col_min_a, col_min_b)
        col_max = max(col_max_a, col_max_b)
        row_min = min(row_min_a, row_min_b)
        row_max = max(row_max_a, row_max_b)
        return [col_min, row_min, col_max, row_max]

    corners_compo_refine = []
    compos_class_refine = []

    for i in range(len(corners_compo_old)):
        a = corners_compo_old[i]
        # broad = draw_bounding_box(org, [a], show=True)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        new_corner = None
        text_area = 0
        for b in corner_text:
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            # get the intersected area
            col_min_s = max(a[0], b[0])
            row_min_s = max(a[1], b[1])
            col_max_s = min(a[2], b[2])
            row_max_s = min(a[3], b[3])
            w = np.maximum(0, col_max_s - col_min_s + 1)
            h = np.maximum(0, row_max_s - row_min_s + 1)
            inter = w * h
            if inter == 0:
                continue

            # calculate IoU
            ioa = inter / area_a
            iob = inter / area_b
            iou = inter / (area_a + area_b - inter)

            # print('ioa:%.3f, iob:%.3f, iou:%.3f' %(ioa, iob, iou))
            # draw_bounding_box(broad, [b], color=(255,0,0), line=2, show=True)

            # text area
            if iou > 0.5 or ioa >= 0.9:
                new_corner = merge_two_corners(a, b)
                break
            text_area += inter

        if new_corner is not None:
            corners_compo_refine.append(new_corner)
            compos_class_refine.append('TextView')
        elif text_area / area_a > 0.4:
            corners_compo_refine.append(corners_compo_old[i])
            compos_class_refine.append('TextView')
        else:
            corners_compo_refine.append(corners_compo_old[i])
            compos_class_refine.append(compos_class_old[i])

    return corners_compo_refine, compos_class_refine


def refine_text(org, corners_text, max_line_gap, min_word_length):
    def refine(bin):
        head = 0
        rear = 0
        gap = 0
        get_word = False
        for i in range(bin.shape[1]):
            # find head
            if not get_word and np.sum(bin[:, i]) != 0:
                head = i
                rear = i
                get_word = True
                continue
            if get_word and np.sum(bin[:, i]) != 0:
                rear = i
                continue
            if get_word and np.sum(bin[:, i]) == 0:
                gap += 1
            if gap > max_line_gap:
                if (rear - head) > min_word_length:
                    corners_text_refine.append((head + col_min, row_min, rear + col_min, row_max))
                gap = 0
                get_word = False

        if get_word and (rear - head) > min_word_length:
            corners_text_refine.append((head + col_min, row_min, rear + col_min, row_max))

    corners_text_refine = []
    pad = 1
    for corner in corners_text:
        (col_min, row_min, col_max, row_max) = corner
        col_min = max(col_min - pad, 0)
        col_max = min(col_max + pad, org.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, org.shape[0])

        if row_max <= row_min or col_max <= col_min:
            continue

        clip = org[row_min:row_max, col_min:col_max]
        clip_bin = pre.preprocess(clip)
        refine(clip_bin)
    return corners_text_refine


def incorporate(img_path, output_root, resize_by_height=None, show=False, write_img=True):
    name = img_path.split('\\')[-1][:-4]
    compo_f = pjoin(output_root, 'cls', name + '.json')
    text_f = pjoin(output_root, 'ocr', name + '.json')
    img, _ = pre.read_img(img_path, resize_by_height)

    compos = json.load(open(compo_f, 'r'))
    texts = json.load(open(text_f, 'r'))
    bbox_compos = []
    class_compos = []
    bbox_text = []
    for compo in compos['compos']:
        bbox_compos.append([compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']])
        class_compos.append(compo['class'])
    for text in texts['compos']:
        bbox_text.append([text['column_min'], text['row_min'], text['column_max'], text['row_max']])

    draw_bounding_box_class(img, bbox_compos, class_compos, show=True)
    bbox_text = refine_text(img, bbox_text, 20, 10)
    corners_compo_new, compos_class_new = nms(img, bbox_compos, class_compos, bbox_text)
    board = draw_bounding_box_class(img, corners_compo_new, compos_class_new)

    output_path = pjoin(output_root, 'merge')
    save_corners_json(pjoin(output_path, name + '.json'), bbox_compos, class_compos)
    if write_img:
        cv2.imwrite(pjoin(output_path, name + '.png'), board)
    if show:
        cv2.imshow('merge', board)
        cv2.waitKey()

    print('Merge Complete and Save to', output_path)
