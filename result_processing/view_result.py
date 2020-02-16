import json
from glob import glob
from os.path import join as pjoin
import cv2

color_map = {'block':(0,255,0), 'compo':(0,0,255)}


def draw_bounding_box(img, corners, classes, resize_height=800,  color_map=color_map, line=2, show=False, write_path=None):
    def resize_by_height(org):
        w_h_ratio = org.shape[1] / org.shape[0]
        resize_w = resize_height * w_h_ratio
        re = cv2.resize(org, (int(resize_w), int(resize_height)))
        return re

    board = resize_by_height(img)
    for i in range(len(corners)):
        board = cv2.rectangle(board, corners[i][0], corners[i][1], color_map[classes[i]], line)
        board = cv2.putText(board, classes[i], (corners[i][0][0]+5, corners[i][0][1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[classes[i]], 2)
    if show:
        cv2.imshow('a', board)
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def view_detect_result_json(reslut_file_root, img_file_root, show=True):
    result_files = glob(pjoin(reslut_file_root, '*_all.json'))
    print(result_files)
    compos_reform = {}
    print('Loading %d detection results' % len(result_files))
    for reslut_file in result_files:
        img_name = reslut_file.split('\\')[-1].split('_')[0]
        compos = json.load(open(reslut_file, 'r'))['compos']
        corners = []
        class_names = []
        for compo in compos:
            corners.append([(compo['column_min'], compo['row_min']), (compo['column_max'], compo['row_max'])])
            class_names.append(compo['class'])

            if show:
                img = cv2.imread(pjoin(img_file_root, img_name + '.jpg'))
                draw_bounding_box(img, corners, class_names, show=True)
    return compos_reform


view_detect_result_json('data\\output\\ip', 'data\\input')
