from tqdm import tqdm
import json
import cv2
from os.path import join as pjoin

from config.CONFIG_UIED import Config
C = Config()


def draw_bounding_box_class(org, components, color=C.COLOR, line=2, show=False, write_path=None):
    """
    Draw bounding box of components with their classes on the original image
    :param org: original image
    :param components: bbox [(column_min, row_min, column_max, row_max)]
                    -> top_left: (column_min, row_min)
                    -> bottom_right: (column_max, row_max)
    :param color_map: colors mapping to different components
    :param line: line thickness
    :param compo_class: classes matching the corners of components
    :param show: show or not
    :return: labeled image
    """
    board = org.copy()
    bboxes = components['bboxes']
    categories = components['categories']
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        category = categories[i]
        board = cv2.rectangle(board, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[C.CLASS_MAP[str(category)]], line)
        board = cv2.putText(board, C.CLASS_MAP[str(category)], (bbox[0]+5, bbox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[C.CLASS_MAP[str(category)]], 2)
    if show:
        cv2.imshow('a', cv2.resize(board, (500, 1000)))
        cv2.waitKey(0)
    if write_path is not None:
        cv2.imwrite(write_path, board)
    return board


def load_ground_truth_json(gt_file, no_text=True):
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
    annots = data['annotations']
    compos = {}
    print('Loading %d ground truth' % len(annots))
    for annot in tqdm(annots):
        img_name, size = get_img_by_id(annot['image_id'])
        if no_text and int(annot['category_id']) == 14:
            compos[img_name] = {'bboxes': [], 'categories': [], 'size': size}
            continue
        if img_name not in compos:
            compos[img_name] = {'bboxes': [cvt_bbox(annot['bbox'])], 'categories': [annot['category_id']], 'size':size}
        else:
            compos[img_name]['bboxes'].append(cvt_bbox(annot['bbox']))
            compos[img_name]['categories'].append(annot['category_id'])
    return compos


def view_gt_all(gt, img_root):
    for img_id in gt:
        compos = gt[img_id]
        img = cv2.imread(pjoin(img_root, img_id + '.jpg'))
        print(pjoin(img_root, img_id + '.jpg'))
        draw_bounding_box_class(img, compos, show=True)


def view_gt_single(gt, img_root, img_id):
    img_id = str(img_id)
    compos = gt[img_id]
    img = cv2.imread(pjoin(img_root, img_id + '.jpg'))
    print(pjoin(img_root, img_id + '.jpg'))
    draw_bounding_box_class(img, compos, show=True)


gt = load_ground_truth_json('E:\\Mulong\\Datasets\\rico\\instances_test.json', no_text=False)
# view_gt_all(gt, 'E:\\Mulong\\Datasets\\rico\\combined')
view_gt_single(gt, 'E:\\Mulong\\Datasets\\rico\\combined', 670)
