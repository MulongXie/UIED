import glob
from os.path import join as pjoin, exists
import time

import ocr_east as ocr
import ip
import merge

from CONFIG import Config

# choose functionality
is_ocr = True
is_ip = True
is_merge = True
# initialization
is_clip = False
C = Config()
C.build_output_folders(is_clip)
resize_by_height = 600

# set input root directory and sort all images by their indices
input_paths_img = glob.glob(pjoin(C.ROOT_INPUT, '*.jpg'))
input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index
# set the range of target inputs' indices
start_index = 24
end_index = 50
for input_path_img in input_paths_img:
    index = input_path_img.split('\\')[-1][:-4]
    if int(index) < start_index:
        continue
    if int(index) > end_index:
        break

    # *** start processing ***
    start = time.clock()

    if is_ocr:
        ocr.east(input_path_img, C.ROOT_OCR, resize_by_height)
    if is_ip:
        ip.compo_detection(input_path_img, C.ROOT_IP, resize_by_height)
    if is_merge:
        merge.incorporate(input_path_img, C.ROOT_OCR, C.ROOT_IP, C.ROOT_MERGE, resize_by_height, is_clip)

    print('*** Total Time Taken:%.3f s ***\n' % (time.clock() - start))
