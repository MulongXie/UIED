import json
import cv2
import numpy as np
from os.path import join as pjoin
import os
import time
import shutil

from detect_merge.Element import Element


def show_elements(org_img, eles, show=False, win_name='element', wait_key=0, shown_resize=None, line=2):
    color_map = {'Text':(0, 0, 255), 'Compo':(0, 255, 0), 'Block':(255, 0, 0)}
    img = org_img.copy()
    for ele in eles:
        color = color_map[ele.category]
        ele.visualize_element(img, color, line)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        cv2.imshow(win_name, img_resize)
        cv2.waitKey(wait_key)
    return img_resize


def save_elements(output_dir, background, elements, img_shape):
    components = {'compos': [], 'img_shape': img_shape}
    clip_dir = pjoin(output_dir, 'clips')
    if background is not None: components['compos'].append(background)

    for i, ele in enumerate(elements):
        corner = ele.put_bbox()
        c = {'id': i, 'class': ele.category,
             'height': ele.height, 'width': ele.width,
             'column_min': corner[0], 'row_min': corner[1], 'column_max': corner[2], 'row_max': corner[3]}
        c['clip_path'] = pjoin(clip_dir, c['class'], str(i) + '.jpg')
        components['compos'].append(c)

    json.dump(components, open(pjoin(output_dir, 'compo.json'), 'w'), indent=4)
    return components['compos']


def refine_elements(compos, texts):
    '''
    1. remove compos contained in text
    2. remove compos containing text area that's too large
    3. store text in a compo if it's contained by the compo as the compo's text child element
    '''
    elements = []
    for compo in compos:
        is_valid = True
        text_area = 0
        contained_texts = []
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(text, bias=2)
            if inter > 0:
                if ioa >= 0.8:
                    is_valid = False
                    break
                text_area += inter
                # the text is contained in the non-text compo
                if iob >= 0.8:
                    contained_texts.append(text)
        if is_valid and text_area / compo.area < 0.8:
            for t in contained_texts:
                t.is_child = True
            compo.children += contained_texts
            elements.append(compo)

    for text in texts:
        if not text.is_child:
            elements.append(text)
    return elements


def dissemble_clip_img_fill(clip_root, org, compos):
    def most_pix_around(pad=6, offset=2):
        '''
        determine the filled background color according to the most surrounding pixel
        '''
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
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)

    bkg = org.copy()
    cls_dirs = []
    for compo in compos:
        cls = compo['class']
        if cls == 'Background':
            compo['path'] = pjoin(clip_root, 'bkg.png')
            continue
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo['id']) + '.jpg')
        compo['path'] = c_path
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)

        col_min, row_min, col_max, row_max = compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']
        cv2.imwrite(c_path, org[row_min:row_max, col_min:col_max])
        # Fill up the background area
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), most_pix_around(), -1)
    cv2.imwrite(pjoin(clip_root, 'bkg.png'), bkg)


def merge(img_path, compo_path, text_path, output_root, show=False, wait_key=0):
    compo_json = json.load(open(compo_path, 'r'))
    text_json = json.load(open(text_path, 'r'))

    # load text and non-text compo
    background = None
    compos = []
    for compo in compo_json['compos']:
        if compo['class'] == 'Background':
            background = compo
            continue
        element = Element((compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']),
                          compo['class'])
        compos.append(element)
    texts = []
    for text in text_json['texts']:
        element = Element((text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text')
        texts.append(element)
    if compo_json['img_shape'] != text_json['img_shape']:
        resize_ratio = compo_json['img_shape'][0] / text_json['img_shape'][0]
        for text in texts:
            text.resize(resize_ratio)

    # check the original detected elements
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (compo_json['img_shape'][1], compo_json['img_shape'][0]))
    show_elements(img_resize, texts + compos, show=show, win_name='element', wait_key=wait_key)

    # refine elements by removing some invalid compo and store contained texts as children elements
    elements = refine_elements(compos, texts)
    board = show_elements(img_resize, elements, show=show, win_name='valid compos', wait_key=wait_key)

    # save all merged elements, clips and blank background
    compos_json = save_elements(output_root, background, elements, img_resize.shape)
    dissemble_clip_img_fill(pjoin(output_root, 'clips'), img_resize, compos_json)
    cv2.imwrite(pjoin(output_root, 'result.jpg'), board)

    print('Merge Complete and Save to', pjoin(output_root, 'result.jpg'))
    print(time.ctime(), '\n')
    if show: cv2.destroyAllWindows()


merge('../data/input/2.jpg', '../data/output/ip/2.json', '../data/output/ocr/2.json', '../data/output', show=True)