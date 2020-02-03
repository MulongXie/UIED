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

# set input root directory and sort all images by their indices
input_paths_img = glob.glob(pjoin(C.ROOT_INPUT, '*.png'))
input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

# set the range of target inputs' indices
start_index = 38132  # 37657
end_index = 70000
img_section = (1500, 1500)  # selected img section, height and width

for input_path_img in input_paths_img:
    index = input_path_img.split('\\')[-1][:-4]
    if int(index) < start_index:
        continue
    if int(index) > end_index:
        break

    # *** start processing ***
    start = time.clock()

    # set output paths
    # for image detection
    label_compo = pjoin(C.ROOT_LABEL_UIED, index + '.json')
    img_uied_drawn = pjoin(C.ROOT_IMG_DRAWN_UIED, index + '.png')
    img_uied_grad = pjoin(C.ROOT_IMG_GRADIENT_UIED, index + '.png')
    # for text recognition (ctpn)
    label_text = pjoin(C.ROOT_LABEL_CTPN, index + '.txt')
    img_ctpn_drawn = pjoin(C.ROOT_IMG_DRAWN_CTPN, index + '.png')
    # for incorporated results
    img_merge = pjoin(C.ROOT_IMG_MERGE, index + '.png')

    try:
        if is_ctpn:
            ocr.ctpn(input_path_img, label_text, img_ctpn_drawn, img_section)
        if is_uied:
            ui.uied(input_path_img, label_compo, img_uied_drawn, img_uied_grad, img_section)
        if is_merge:
            merge.incorporate(input_path_img, label_compo, label_text, img_merge, img_section, is_clip)
    except:
        print("Bad Img")

    print('Time Taken:%.3f s\n' % (time.clock() - start))
