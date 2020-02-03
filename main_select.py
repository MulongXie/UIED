import glob
from os.path import join as pjoin, exists
import time

import ocr
import ui
import merge

from CONFIG import Config
# create output directory if inexistent
is_clip = False
C = Config()
C.build_output_folders(is_clip)

# choose functionality
is_ctpn = True
is_uied = True
is_merge = True
img_section = (1500, 1500)  # selected img section, height and width

annotations = open('label.txt', 'r')

start = 1
index = 0
for line in annotations.readlines():
    if index < start:
        index += 1
        continue
    input_path_img = line.split()[0]
    print(input_path_img)

    # *** start processing ***
    start = time.clock()

    # set output paths
    # for image detection
    label_compo = pjoin(C.ROOT_LABEL_UIED, str(index) + '.json')
    img_uied_drawn = pjoin(C.ROOT_IMG_DRAWN_UIED, str(index) + '.png')
    img_uied_grad = pjoin(C.ROOT_IMG_GRADIENT_UIED, str(index) + '.png')
    # for text recognition (ctpn)
    label_text = pjoin(C.ROOT_LABEL_CTPN, str(index) + '.txt')
    img_ctpn_drawn = pjoin(C.ROOT_IMG_DRAWN_CTPN, str(index) + '.png')
    # for incorporated results
    img_merge = pjoin(C.ROOT_IMG_MERGE, str(index) + '.png')
    label_merge = pjoin(C.ROOT_OUTPUT, 'label.txt')

    if is_ctpn:
        ocr.ctpn(input_path_img, label_text, img_ctpn_drawn, img_section)
    if is_uied:
        ui.uied(input_path_img, label_compo, img_uied_drawn, img_uied_grad, img_section)
    if is_merge:
        merge.incorporate(input_path_img, label_compo, label_text, img_merge, label_merge, img_section, is_clip)

    print('%d Time Taken:%.3f s\n' % (index, time.clock() - start))
    index += 1

