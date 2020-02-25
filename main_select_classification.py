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
    data = json.load(open('E:\\Mulong\\Datasets\\rico\\instances_test_org.json', 'r'))
    input_paths_img = [pjoin(input_root, img['file_name'].split('/')[-1]) for img in data['images']]
    input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

    # switch of the classification func
    is_clf = False
    if is_clf:
        from Resnet import ResClassifier
        classifier = ResClassifier()
    else:
        classifier = None

    # set the range of target inputs' indices
    num = 0
    start_index = 669
    end_index = 100000
    for input_path_img in input_paths_img:
        index = input_path_img.split('\\')[-1][:-4]
        if int(index) < start_index:
            continue
        if int(index) > end_index:
            break
        ip.compo_detection(input_path_img, output_root, num, show=True,
                           resize_by_height=resize_by_height, classifier=classifier)
        num += 1
