import multiprocessing
import glob
import time
import json
from tqdm import tqdm
from os.path import join as pjoin, exists

import ip_region_proposal as ip
from CONFIG import Config

if __name__ == '__main__':
    # initialization
    C = Config()
    resize_by_height = 800
    input_root = C.ROOT_INPUT
    output_root = C.ROOT_OUTPUT

    # set input root directory and sort all images by their indices
    data = json.load(open('E:\\Mulong\\Datasets\\rico\\instances_test.json', 'r'))
    input_paths_img = [pjoin(input_root, img['file_name'].split('/')[-1]) for img in data['images']]
    input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

    is_ip = False
    is_ocr = False
    is_merge = True

    # switch of the classification func
    classifier = None
    if is_ip:
        is_clf = True
        if is_clf:
            classifier = {}
            from CNN import CNN
            classifier['Image'] = CNN('Image')
            classifier['Elements'] = CNN('Elements')
            classifier['Noise'] = CNN('Noise')
    # set the range of target inputs' indices
    num = 0
    start_index = 30800  # 61728
    end_index = 100000
    for input_path_img in input_paths_img:
        index = input_path_img.split('\\')[-1][:-4]
        if int(index) < start_index:
            continue
        if int(index) > end_index:
            break

        if is_ocr:
            import ocr_east as ocr
            ocr.east(input_path_img, output_root, resize_by_height=None, show=False, write_img=True)

        if is_ip:
            ip.compo_detection(input_path_img, output_root, num, resize_by_height=resize_by_height, show=True, classifier=classifier)

        if is_merge:
            import merge
            compo_path = pjoin(output_root, 'ip', str(index) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(index) + '.json')
            merge.incorporate(input_path_img, compo_path, ocr_path, output_root, resize_by_height=resize_by_height, show=True, write_img=True)

        num += 1
