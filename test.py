from glob import glob
from os.path import join as pjoin
class_map = ['button', 'input', 'select', 'search', 'list', 'img', 'block', 'text', 'icon']


def read_label(file_name):
    '''
    :return: {list of [[col_min, row_min, col_max, row_max]], list of [class id]
    '''

    def is_bottom_or_top(bbox):
        if bbox[1] < 100 and (bbox[1] + bbox[3]) < 100 or\
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
    return {file_name.split('_')[0]: {'bboxes':bboxes, 'categories':categories}}


def load_detect_result(input_root):
    compos = []
    label_paths = glob(pjoin(input_root, '*.txt'))
    for label_path in label_paths:
        compo = read_label(label_path)
        print(compo)
        compos.append(compo)
    return compos


load_detect_result("E:\\Mulong\\Result\\rico\\merge")