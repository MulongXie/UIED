import json
import numpy as np

class_map = ['button', 'input', 'select', 'search', 'list', 'img', 'block', 'text', 'icon']


def read_detect_result(file_name):
    '''
    :return: {list of [[col_min, row_min, col_max, row_max]], list of [class id]
    '''
    file = open(file_name, 'r')
    bboxes = []
    categories = []
    for l in file.readlines():
        labels = l.split()[1:]
        for label in labels:
            label = label.split(',')
            bboxes.append([int(b) for b in label[:-1]])
            categories.append(class_map[int(label[-1])])

    return {file_name.split('_')[0]: {'bboxes':bboxes, 'categories':categories}}


def read_ground_truth():
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

    data = json.load(open('instances_test.json', 'r'))

    images = data['images']
    annots = data['annotations']

    compos = {}
    for annot in annots:
        img_name, size = get_img_by_id(annot['image_id'])
        if img_name not in compos:
            compos[img_name] = {'bboxes': [annot['bbox']], 'categories': [annot['category_id']], 'size':size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def match(d_bbox, gt_bboxes, matched):
    '''
    :param matched: mark if the ground truth component is matched
    :param d_bbox: [col_min, row_min, col_max, row_max]
    :param gt_bboxes: list of ground truth [[col_min, row_min, col_max, row_max]]
    :return: Boolean: if IOU large enough or detected box is contained by ground truth
    '''
    area_d = (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])
    for i, gt_bbox in enumerate(gt_bboxes):
        if matched[i] == 0:
            continue
        area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[0])
        col_min = max(d_bbox[0], gt_bbox[0])
        row_min = max(d_bbox[1], gt_bbox[1])
        col_max = min(d_bbox[2], gt_bbox[2])
        row_max = min(d_bbox[3], gt_bbox[3])
        # if not intersected, w or h should be 0
        w = max(0, col_max - col_min)
        h = max(0, row_max - row_min)
        area_inter = w*h
        if area_inter == 0:
            continue

        iod = area_inter / area_d
        iou = area_inter / (area_d + area_gt)
        # the interaction is d itself, so d is contained in gt, considered as correct detection
        if iod == area_d and area_d / area_gt > 0.1:
            matched[i] = 0
            return True
        if iou > 0.5:
            return True
    return False


def resize_label(bboxes, d_height, gt_height, bias=10):
    bboxes_new = []
    scale = gt_height/d_height
    for bbox in bboxes:
        bbox = [b * scale + bias for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def eval(detection, ground_truth):
    TP, FP, FN = 0, 0, 0
    for image in detection:
        d_compo = detection[image]
        gt_compo = ground_truth[image]
        matched = np.ones(len(gt_compo['bboxes']), dtype=int)
        for d_bbox in d_compo['bboxes']:
            if match(d_bbox, gt_compo['bboxes'], matched):
                TP += 1
            else:
                FP += 1
        FN += sum(matched)

    print(TP, FP, FN)


gt = read_ground_truth()
detect = read_detect_result('472_merged.txt')
eval(detect, gt)
