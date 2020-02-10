import json
import numpy as np
import cv2
from glob import glob
from os.path import join as pjoin
from tqdm import tqdm

class_map = ['button', 'input', 'select', 'search', 'list', 'img', 'block', 'text', 'icon']


def resize_label(bboxes, d_height, gt_height, bias=10):
    bboxes_new = []
    scale = gt_height/d_height
    for bbox in bboxes:
        bbox = [int(b * scale + bias) for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def draw_bounding_box(org, corners, color=(0, 255, 0), line=2, show=False):
    """
    Draw bounding box of components on the original image
    :param org: original image
    :param corners: [(top_left, bottom_right)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color: line color
    :param line: line thickness
    :param show: show or not
    :return: labeled image
    """
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color, line)
    if show:
        cv2.imshow('a', cv2.resize(board, (300, 600)))
        cv2.waitKey(0)
    return board


def load_detect_result_txt(input_root):
    def read_label(file_name):
        '''
        :return: {list of [[col_min, row_min, col_max, row_max]], list of [class id]
        '''

        def is_bottom_or_top(bbox):
            if bbox[1] < 100 and (bbox[1] + bbox[3]) < 100 or \
                    bbox[1] > 500 and (bbox[1] + bbox[3]) > 500:
                return True
            return False

        file = open(file_name, 'r')
        bboxes = []
        categories = []
        for l in file.readlines():
            labels = l.split()[1:]
            for label in labels:
                label = label.split(',')
                bbox = [int(b) for b in label[:-1]]
                if not is_bottom_or_top(bbox):
                    bboxes.append(bbox)
                    categories.append(class_map[int(label[-1])])
        index = file_name.split('\\')[-1].split('_')[0]
        return index, {'bboxes': bboxes, 'categories': categories}

    compos = {}
    label_paths = glob(pjoin(input_root, '*.txt'))
    print('Loading %d detection results' % len(label_paths))
    for label_path in tqdm(label_paths):
        index, bboxes = read_label(label_path)
        compos[index] = bboxes
    return compos


def load_detect_result_json(reslut_file, gt_file):
    def get_img_by_id(img_id):
        for image in images:
            if image['id'] == img_id:
                return image['file_name'].split('/')[-1][:-4], (image['height'], image['width'])

    def cvt_bbox(bbox):
        '''
        :param bbox: [x,y,width,height]
        :return: [col_min, row_min, col_max, row_max]
        '''
        bbox = [int(b) for b in bbox]
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    data = json.load(open(gt_file, 'r'))
    images = data['images']
    compos = {}
    annots = json.load(open(reslut_file, 'r'))
    print('Loading %d detection results' % len(annots))
    for annot in tqdm(annots):
        img_name, size = get_img_by_id(annot['image_id'])
        if img_name not in compos:
            compos[img_name] = {'bboxes': [cvt_bbox(annot['bbox'])], 'categories': [annot['category_id']], 'size': size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def load_ground_truth_json(annotation_file):
    def get_img_by_id(img_id):
        for image in images:
            if image['id'] == img_id:
                return image['file_name'].split('/')[-1][:-4], (image['height'], image['width'])

    def cvt_bbox(bbox):
        '''
        :param bbox: [x,y,width,height]
        :return: [col_min, row_min, col_max, row_max]
        '''
        bbox = [int(b) for b in bbox]
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    data = json.load(open(annotation_file, 'r'))
    images = data['images']
    annots = data['annotations']
    compos = {}
    print('Loading %d ground truth' % len(annots))
    for annot in tqdm(annots):
        img_name, size = get_img_by_id(annot['image_id'])
        if img_name not in compos:
            compos[img_name] = {'bboxes': [cvt_bbox(annot['bbox'])], 'categories': [annot['category_id']], 'size':size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def eval(detection, ground_truth, org_root, show=True):
    def match(org, d_bbox, gt_bboxes, matched):
        '''
        :param matched: mark if the ground truth component is matched
        :param d_bbox: [col_min, row_min, col_max, row_max]
        :param gt_bboxes: list of ground truth [[col_min, row_min, col_max, row_max]]
        :return: Boolean: if IOU large enough or detected box is contained by ground truth
        '''
        area_d = (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])
        broad = draw_bounding_box(org, [d_bbox], color=(0, 0, 255))
        for i, gt_bbox in enumerate(gt_bboxes):
            if matched[i] == 0:
                continue
            area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            col_min = max(d_bbox[0], gt_bbox[0])
            row_min = max(d_bbox[1], gt_bbox[1])
            col_max = min(d_bbox[2], gt_bbox[2])
            row_max = min(d_bbox[3], gt_bbox[3])
            # if not intersected, area intersection should be 0
            w = max(0, col_max - col_min)
            h = max(0, row_max - row_min)
            area_inter = w * h
            if area_inter == 0:
                continue
            iod = area_inter / area_d
            iou = area_inter / (area_d + area_gt - area_inter)

            if show:
                print("IoDetection: %.3f, IoU: %.3f" % (iod, iou))
                draw_bounding_box(broad, [gt_bbox], color=(0, 255, 0), show=True)

            # the interaction is d itself, so d is contained in gt, considered as correct detection
            if iod == area_d and area_d / area_gt > 0.1:
                matched[i] = 0
                return True
            if iou > 0.5:
                return True
        return False

    amount = len(detection)
    TP, FP, FN = 0, 0, 0
    for i, image_id in enumerate(tqdm(detection)):
        d_compos = detection[image_id]
        image_id = image_id.split('.')[0]
        img = cv2.imread(pjoin(org_root, image_id + '.jpg'))
        gt_compos = ground_truth[image_id]
        d_compos['bboxes'] = resize_label(d_compos['bboxes'], 600, gt_compos['size'][0])
        # mark matched bboxes
        matched = np.ones(len(gt_compos['bboxes']), dtype=int)
        for d_bbox in d_compos['bboxes']:
            if match(img, d_bbox, gt_compos['bboxes'], matched):
                TP += 1
            else:
                FP += 1
        FN += sum(matched)

        if i % 100 == 0:
            precesion = TP / (TP+FP)
            recall = TP / (TP+FN)
            print('[%d/%d] TP:%d, FP:%d, FN:%d, Precesion:%.3f, Recall:%.3f'
                  % (i, amount, TP, FP, FN, precesion, recall))


detect = load_detect_result_txt('E:\\Mulong\\Result\\rico_remaui')
# detect = load_detect_result_json('E:\Temp\detections_val_results.json', 'E:/Mulong/Datasets/rico/instances_val.json')
gt = load_ground_truth_json('E:/Mulong/Datasets/rico/instances_test.json')
eval(detect, gt, 'E:\\Mulong\\Datasets\\rico\\combined', show=False)
