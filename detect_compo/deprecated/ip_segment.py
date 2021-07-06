import cv2
import numpy as np
import shutil
import os
from os.path import join as pjoin


def segment_img(org, segment_size, output_path, overlap=100):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    height, width = np.shape(org)[0], np.shape(org)[1]
    top = 0
    bottom = segment_size
    segment_no = 0
    while top < height and bottom < height:
        segment = org[top:bottom]
        cv2.imwrite(os.path.join(output_path, str(segment_no) + '.png'), segment)
        segment_no += 1
        top += segment_size - overlap
        bottom = bottom + segment_size - overlap if bottom + segment_size - overlap <= height else height


def clipping(img, components, pad=0, show=False):
    """
    :param adjust: shrink(negative) or expand(positive) the bounding box
    :param img: original image
    :param corners: ((column_min, row_min),(column_max, row_max))
    :return: list of clipping images
    """
    clips = []
    for component in components:
        clip = component.compo_clipping(img, pad=pad)
        clips.append(clip)
        if show:
            cv2.imshow('clipping', clip)
            cv2.waitKey()
    return clips


def dissemble_clip_img_hollow(clip_root, org, compos):
    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)
    cls_dirs = []

    bkg = org.copy()
    hollow_out = np.ones(bkg.shape[:2], dtype=np.uint8) * 255
    for compo in compos:
        cls = compo.category
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo.id) + '.jpg')
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)
        clip = compo.compo_clipping(org)
        cv2.imwrite(c_path, clip)

        col_min, row_min, col_max, row_max = compo.put_bbox()
        hollow_out[row_min: row_max, col_min: col_max] = 0

    bkg = cv2.merge((bkg, hollow_out))
    cv2.imwrite(os.path.join(clip_root, 'bkg.png'), bkg)


def dissemble_clip_img_fill(clip_root, org, compos, flag='most'):

    def average_pix_around(pad=6, offset=3):
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1

        average = []
        for i in range(3):
            avg_up = np.average(org[up:row_min - offset, left:right, i])
            avg_bot = np.average(org[row_max + offset:bottom, left:right, i])
            avg_left = np.average(org[up:bottom, left:col_min - offset, i])
            avg_right = np.average(org[up:bottom, col_max + offset:right, i])
            average.append(int((avg_up + avg_bot + avg_left + avg_right)/4))
        return average

    def most_pix_around(pad=6, offset=2):
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1

        most = []
        for i in range(3):
            val = np.concatenate((org[up:row_min - offset, left:right, i].flatten(),
                            org[row_max + offset:bottom, left:right, i].flatten(),
                            org[up:bottom, left:col_min - offset, i].flatten(),
                            org[up:bottom, col_max + offset:right, i].flatten()))
            # print(val)
            # print(np.argmax(np.bincount(val)))
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)
    cls_dirs = []

    bkg = org.copy()
    for compo in compos:
        cls = compo.category
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo.id) + '.jpg')
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)
        clip = compo.compo_clipping(org)
        cv2.imwrite(c_path, clip)

        col_min, row_min, col_max, row_max = compo.put_bbox()
        if flag == 'average':
            color = average_pix_around()
        elif flag == 'most':
            color = most_pix_around()
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), color, -1)

    cv2.imwrite(os.path.join(clip_root, 'bkg.png'), bkg)
