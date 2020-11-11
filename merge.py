import json
import cv2
import numpy as np
from os.path import join as pjoin
import os
import time
from random import randint as rint

from utils.util_merge import *
from config.CONFIG import Config
from utils.Element import Element
C = Config()

def reclassify_text_by_ocr(org, compos, texts):
    compos_new = []
    for i, compo in enumerate(compos):
        # broad = draw_bounding_box(org, [compo], show=True)
        new_compo = None
        text_area = 0
        for j, text in enumerate(texts):
            # get the intersected area
            inter = compo.calc_intersection_area(text)
            if inter == 0:
                continue

            # calculate IoU
            ioa = inter / compo.area
            iob = inter / text.area
            iou = inter / (compo.area + text.area - inter)

            # print('ioa:%.3f, iob:%.3f, iou:%.3f' %(ioa, iob, iou))
            # draw_bounding_box(broad, [text], color=(255,0,0), line=2, show=True)

            # text area
            if ioa >= 0.68 or iou > 0.55:
                new_compo = compo.element_merge(text, new_element=True, new_category='Text')
                texts[j] = new_compo
                break
            text_area += inter

        # print("Text area ratio:%.3f" % (text_area / compo.area))
        if new_compo is not None:
            compos_new.append(new_compo)
        elif text_area / compo.area > 0.4:
            compo.category = 'Text'
            compos_new.append(compo)
        else:
            compos_new.append(compo)
    return compos_new


def merge_intersected_compos(org, compos, max_gap=(0, 0), merge_class=None):
    changed = False
    new_compos = []
    for i in range(len(compos)):
        if merge_class is not None and compos[i].category != merge_class:
            new_compos.append(compos[i])
            continue
        merged = False
        cur_compo = compos[i]
        for j in range(len(new_compos)):
            if merge_class is not None and new_compos[j].category != merge_class:
                continue
            relation = cur_compo.element_relation(new_compos[j], max_gap)
            # print(relation)
            # draw_bounding_box(org, [cur_compo, new_compos[j]], name='b-merge', show=True)
            if relation != 0:
                new_compos[j].element_merge(cur_compo)
                cur_compo = new_compos[j]
                # draw_bounding_box(org, [new_compos[j]], name='a-merge', show=True)
                merged = True
                changed = True
                # break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_intersected_compos(org, new_compos, max_gap, merge_class)


def rm_compos_in_text(compos):
    mark = np.zeros(len(compos))
    for i, c1 in enumerate(compos):
        if c1.category != 'Text':
            continue
        for j, c2 in enumerate(compos):
            if c2.category == 'Text' or mark[j] != 0:
                continue
            if c1.element_relation(c2) != 0:
                c1.element_merge(c2)
                mark[j] = 1

    new_compos = []
    for i, m in enumerate(mark):
        if m == 0:
            new_compos.append(compos[i])
    return new_compos


def incorporate(img_path, compo_path, text_path, output_root, params,
                resize_by_height=None, show=False, wait_key=0):
    org = cv2.imread(img_path)

    compos = []
    texts = []

    background = None
    for compo in json.load(open(compo_path, 'r'))['compos']:
        if compo['class'] == 'Background':
            background = compo
            continue
        element = Element((compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']), compo['class'])
        compos.append(element)
    for text in json.load(open(text_path, 'r'))['compos']:
        element = Element((text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text')
        texts.append(element)

    org_resize = resize_img_by_height(org, resize_by_height)
    draw_bounding_box_class(org_resize, compos, show=show, name='ip', wait_key=wait_key)
    draw_bounding_box(org_resize, texts, show=show, name='ocr', wait_key=wait_key)

    compos_merged = reclassify_text_by_ocr(org_resize, compos, texts)
    # compos_merged = merge_redundant_corner(org_resize, compos_merged)
    # draw_bounding_box_class(org_resize, compos_merged, name='text', show=show, wait_key=wait_key)

    # merge words as line
    compos_merged = merge_intersected_compos(org_resize, compos_merged, max_gap=(params['max-word-inline-gap'], 0), merge_class='Text')
    draw_bounding_box_class(org_resize, compos_merged, name='merged line', show=show, wait_key=wait_key)
    # merge lines as paragraph
    compos_merged = merge_intersected_compos(org_resize, compos_merged, max_gap=(0, params['max-line-gap']), merge_class='Text')
    # draw_bounding_box_class(org_resize, compos_merged, name='merged paragraph', show=show)
    # clean compos intersected with paragraphs
    compos_merged = rm_compos_in_text(compos_merged)
    board = draw_bounding_box_class(org_resize, compos_merged, name='merged paragraph', is_return=True, show=show, wait_key=wait_key)

    # draw_bounding_box_non_text(org_resize, compos_merged, org_shape=org.shape, show=show)
    compos_json = save_corners_json(output_root, background, compos_merged, org_resize.shape)
    dissemble_clip_img_fill(pjoin(output_root, 'clips'), org_resize, compos_json)
    cv2.imwrite(pjoin(output_root, 'result.jpg'), board)

    print('Merge Complete and Save to', pjoin(output_root, 'result.jpg'))
    print(time.ctime(), '\n')
    # if show:
    #     cv2.destroyAllWindows()

