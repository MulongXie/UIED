import json
import cv2
import numpy as np
from os.path import join as pjoin
import os

import lib_ip.ip_preprocessing as pre
from config.CONFIG import Config
C = Config()
compo_index = {'img':0, 'text':0, 'button':0, 'input':0, 'icon':0}


def draw_bounding_box_class(org, corners, compo_class, color_map=C.COLOR, line=3, show=False, name='img'):
    board = org.copy()
    for i in range(len(corners)):
        if compo_class[i] == 'text':
            continue
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color_map[compo_class[i]], line)
        board = cv2.putText(board, compo_class[i], (corners[i][0]+5, corners[i][1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[compo_class[i]], 2)
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


def save_clipping(org, corners, compo_classes, compo_index, output_root=C.ROOT_IMG_COMPONENT):
    if output_root is None:
        output_root = C.ROOT_IMG_COMPONENT
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    pad = 1
    for i in range(len(corners)):
        compo = compo_classes[i]
        (col_min, row_min, col_max, row_max) = corners[i]
        col_min = max(col_min - pad, 0)
        col_max = min(col_max + pad, org.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, org.shape[0])

        # if component type already exists, index increase by 1, otherwise add this type
        compo_path = pjoin(output_root, compo)
        if not os.path.exists(compo_path):
            os.mkdir(compo_path)
        if compo_classes[i] not in compo_index:
            compo_index[compo_classes[i]] = 0
        else:
            compo_index[compo_classes[i]] += 1
        clip = org[row_min:row_max, col_min:col_max]
        cv2.imwrite(pjoin(compo_path, str(compo_index[compo_classes[i]]) + '.png'), clip)


def save_label_txt(img_path, compo_corners, compo_class, label_txt_path):
    f = open(label_txt_path, 'a')
    label_txt = img_path + ' '
    for i in range(len(compo_corners)):
        if compo_class[i] == 'text':
            continue
        label_txt += ','.join([str(c) for c in compo_corners[i]]) + ',' + str(C.class_index[compo_class[i]]) + ' '
    label_txt += '\n'
    f.write(label_txt)


def nms(org, corners_compo_old, compos_class_old, corner_text):
    corners_compo_refine = []
    compos_class_refine = []

    corner_text = np.array(corner_text)
    for i in range(len(corners_compo_old)):
        # if compos_class_old[i] != 'img':
        #     corners_compo_refine.append(corners_compo_old[i])
        #     compos_class_refine.append(compos_class_old[i])
        #     continue

        a = corners_compo_old[i]
        # broad = draw_bounding_box(org, [a], show=True)
        noise = False
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_text = 0
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

            # print(ioa, iob)
            # draw_bounding_box(broad, [b], color=(255,0,0), line=2, show=True)

            if compos_class_old[i] == 'img':
                # sum up all text area in a img
                # if iob > 0.8:
                area_text += inter
                # loose threshold for img
                if ioa > 0.38:
                    noise = True
                    break
            else:
                # tight threshold for other components
                if ioa > 0.8:
                    noise = True
                    break
        # print(area_text / area_a)
        # check if img is text paragraph
        if compos_class_old[i] == 'img' and area_text / area_a > 0.4:
            noise = True

        if not noise:
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


def incorporate(img_path, label_text, labels_compo, label_merge, resize_by_height=600, is_clip=False, clip_path=None):
    name = img_path.split('\\')[-1][:-4]
    compo_path = pjoin(labels_compo, name + '_ip.json')
    text_path = pjoin(label_text, name + '_ocr.txt')

    img, _ = pre.read_img(img_path, resize_by_height)
    compo_f = open(compo_path, 'r')
    text_f = open(text_path, 'r')

    compos = json.load(compo_f)
    corners_compo = []
    compos_class = []
    corners_text = []
    for compo in compos['compos']:
        corners_compo.append([compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']])
        compos_class.append(compo['class'])
    for line in text_f.readlines():
        if len(line) > 1:
            corners_text.append([int(c) for c in line[:-1].split(',')])

    corners_text = refine_text(img, corners_text, 20, 10)
    corners_compo_new, compos_class_new = nms(img, corners_compo, compos_class, corners_text)
    board = draw_bounding_box_class(img, corners_compo_new, compos_class_new)

    output_path_label = pjoin(label_merge, name + '_merged.txt')
    output_path_img = pjoin(label_merge, name + '_merged.png')
    save_label_txt(img_path, corners_compo_new, compos_class_new, output_path_label)
    cv2.imwrite(output_path_img, board)

    print('Merge Complete and Save to', output_path_img)

    if is_clip:
        save_clipping(img, corners_compo_new, compos_class_new, compo_index, clip_path)
