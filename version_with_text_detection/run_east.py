import ocr_east as ocr

import multiprocessing
import glob
import time
import json
from tqdm import tqdm
from os.path import join as pjoin, exists

ROOT_INPUT = 'E:\\Mulong\\Datasets\\rico\\combined'
ROOT_OUTPUT = 'E:\\Mulong\\Result\\east'

# set input root directory and sort all images by their indices
data = json.load(open('E:\\Mulong\\Datasets\\rico\\instances_test_org.json', 'r'))
input_paths_img = [pjoin(ROOT_INPUT, img['file_name'].split('/')[-1]) for img in data['images']]
input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

start_index = 9122
end_index = 100000
for i, input_path_img in enumerate(input_paths_img):
    print(i, '/', len(input_paths_img))
    index = input_path_img.split('\\')[-1][:-4]
    if int(index) < start_index:
        continue
    if int(index) > end_index:
        break
    ocr.east(input_path_img, ROOT_OUTPUT, resize_by_height=None, show=True)
