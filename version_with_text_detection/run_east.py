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
data = json.load(open('E:\\Mulong\\Datasets\\rico\\instances_val_notext.json', 'r'))
input_paths_img = [pjoin(ROOT_INPUT, img['file_name'].split('/')[-1]) for img in data['images']]
input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

for input_path_img in input_paths_img:
    ocr.east(input_path_img, ROOT_OUTPUT, resize_by_height=None, show=False)
