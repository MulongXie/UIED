import json
from glob import glob
from os.path import join as pjoin
import cv2
import numpy as np

import lib_ip.ip_segment as seg
import CONFIG
cfg = CONFIG.Config()


def draw_bounding_box(img, corners, classes, resize_height=800, color_map=cfg.COLOR, line=2, show=False, write_path=None):
    def resize_by_height(org):
        w_h_ratio = org.shape[1] / org.shape[0]
        resize_w = resize_height * w_h_ratio
        re = cv2.resize(org, (int(resize_w), int(resize_height)))
        return re

    board = resize_by_height(img)
    for i in range(len(corners)):
        board = cv2.rectangle(board, corners[i][0], corners[i][1], color_map[classes[i]], line)
        board = cv2.putText(board, classes[i], (corners[i][0][0]+5, corners[i][0][1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_map[classes[i]], 2)
    if show:
        cv2.imshow('all', board)
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def view_detect_result_json(reslut_file_root, img_file_root, classifier=None, show=True):
    result_files = glob(pjoin(reslut_file_root, '*.json'))
    result_files = sorted(result_files, key=lambda x: int(x.split('\\')[-1].split('_')[0]))
    print('Loading %d detection results' % len(result_files))
    for reslut_file in result_files:
        start_index = 0
        end_index = 100000
        index = reslut_file.split('\\')[-1].split('_')[0]

        if int(index) < start_index:
            continue
        if int(index) > end_index:
            break

        org = cv2.imread(pjoin(img_file_root, index + '.jpg'))
        print(index)
        compos = json.load(open(reslut_file, 'r'))['compos']
        bboxes = []
        for compo in compos:
            bboxes.append([(compo['column_min'], compo['row_min']), (compo['column_max'], compo['row_max'])])

        if classifier is not None:
            classes = classifier.predict(seg.clipping(org, bboxes))
        else:
            classes = np.full(len(bboxes), 'ImageView')

        if show:
            draw_bounding_box(org, bboxes, classes, show=True)


is_clf = True
if is_clf:
    from Resnet import ResClassifier
    classifier = ResClassifier()
else:
    classifier = None

view_detect_result_json('E:\\Mulong\\Result\\rico\\rico_new_uied\\all_corners',
                        "E:\\Mulong\\Datasets\\rico\\combined",
                        classifier=classifier)
