import cv2
from os.path import join as pjoin
import time

import lib_ip.ip_preprocessing as pre
import lib_ip.ip_draw as draw
import lib_ip.ip_detection as det
import lib_ip.ip_segment as seg
import lib_ip.file_utils as file
import lib_ip.ocr_classify_text as ocr
import lib_ip.ip_detection_utils as util
import lib_ip.block_division as blk
from config.CONFIG_UIED import Config
from config.MODEL_CNN import CNN

cnn = CNN()
cnn.load()


def processing_block(org, binary, blocks_corner, classifier):
    '''
    :param org: original image
    :param binary: binary map of original image
    :param blocks_corner: list of corners of blocks
                        [(top_left, bottom_right)]
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    :param classifier: cnn model
    :return: boundaries of detected components in blocks;
                        [up, bottom, left, right]
                        -> up, bottom: list of [(column_index, min/max row border)]
                        -> left, right: list of [(row_index, min/max column border)]
             corners of detected components in blocks;
             corresponding classes of components;
    '''
    blocks_clip_org = seg.clipping(org, blocks_corner, shrink=3)
    blocks_clip_bin = seg.clipping(binary, blocks_corner, shrink=3)

    all_compos_boundary = []
    all_compos_corner = []
    all_compos_class = []
    for i in range(len(blocks_corner)):
        # *** Substep 1.1 *** pre-processing: get block information -> binarization
        block_corner = blocks_corner[i]
        if blk.block_is_top_or_bottom_bar(blocks_corner[i], org.shape): continue
        block_clip_org = blocks_clip_org[i]
        block_clip_bin = blocks_clip_bin[i]

        # *** Substep 1.2 *** object extraction: extract components boundary -> get bounding box corner
        compos_boundary = det.boundary_detection(block_clip_bin)
        compos_corner = det.get_corner(compos_boundary)

        # *** Substep 1.3 *** classification: clip components -> classify components
        compos_clip = seg.clipping(block_clip_org, compos_corner)
        compos_class = classifier.predict(compos_clip)

        # *** Substep 1.4 *** refining: merge overlapping components -> convert the corners to holistic value in entire image
        compos_corner, compos_class = det.merge_corner(compos_corner, compos_class)
        compos_corner = util.corner_cvt_relative_position(compos_corner, block_corner[0][0], block_corner[0][1])

        if len(compos_boundary) > 0:
            all_compos_boundary += compos_boundary
            all_compos_corner += compos_corner
            all_compos_class += compos_class
    return all_compos_boundary, all_compos_corner, all_compos_class


def processing(org, binary, classifier, inspect_img=False):
    # *** Substep 2.1 *** object detection: get connected areas -> get boundary -> get corners
    compos_boundary = det.boundary_detection(binary)
    compos_corner = det.get_corner(compos_boundary)

    # *** Substep 2.2 *** classification: clip components -> classify components
    compos_clip = seg.clipping(org, compos_corner)
    compos_class = classifier.predict(compos_clip)

    # *** Substep 2.3 *** refining: merge overlapping components -> search components on background image
    compos_corner, compos_class = det.merge_corner(compos_corner, compos_class)
    if inspect_img:
        compos_corner, compos_class = det.compo_on_img(processing, org, binary, classifier, compos_corner, compos_class)
    return compos_boundary, compos_corner, compos_class


def compo_detection(input_img_path, output_root, resize_by_height=600):
    start = time.clock()
    print("Compo Detection for %s" % input_img_path)
    name = input_img_path.split('\\')[-1][:-4]

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary_org = pre.preprocess(org, write_path=pjoin(output_root, name + '_binary.png'))

    # *** Step 2 *** block processing: detect block -> detect components in block
    blocks_corner = blk.block_division(grey, write_path=pjoin(output_root, name + '_block.png'))
    compo_in_blk_boundary, compo_in_blk_corner, compo_in_blk_class = processing_block(org, binary_org, blocks_corner, cnn)

    # *** Step 3 *** non-block processing: erase blocks from binary -> detect left components
    binary_non_block = blk.block_erase(binary_org, blocks_corner)
    compo_non_blk_boundary, compo_non_blk_corner, compo_non_blk_class = processing(org, binary_non_block, cnn, True)

    # *** Step 4 *** merge results
    # compos_boundary = compo_in_blk_boundary + compo_non_blk_boundary
    compos_corner = compo_in_blk_corner + compo_non_blk_corner
    compos_class = compo_in_blk_class + compo_non_blk_class

    # *** Step 5 *** save results: save text label -> save drawn image
    draw.draw_bounding_box_class(org, compos_corner, compos_class, write_path=pjoin(output_root, name + '_ip.png'))
    file.save_corners_json(pjoin(output_root, name + '_ip.json'), compos_corner, compos_class)

    print("[Compo Detection Completed in %.3f s]" % (time.clock() - start))
