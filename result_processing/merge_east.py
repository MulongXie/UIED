import multiprocessing
from glob import glob
import time
import json
from tqdm import tqdm
from os.path import join as pjoin, exists

import merge


input_root = 'E:\\Mulong\\Datasets\\rico\\combined'
output_root = 'E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_cls\\merge'
compo_root = 'E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_cls\\ip'
text_root = 'E:\\Mulong\\Result\\east'

data = json.load(open('E:\\Mulong\\Datasets\\rico\\instances_test.json', 'r'))
input_paths_img = [pjoin(input_root, img['file_name'].split('/')[-1]) for img in data['images']]
input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

# set the range of target inputs' indices
num = 0
start_index = 0
end_index = 100000
for input_path_img in input_paths_img:
    index = input_path_img.split('\\')[-1][:-4]
    if int(index) < start_index:
        continue
    if int(index) > end_index:
        break

    merge.incorporate(input_path_img, compo_root, text_root, output_root, resize_by_height=800, show=False)
