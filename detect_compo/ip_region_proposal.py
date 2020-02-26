import cv2
from os.path import join as pjoin
import time
import numpy as np

import lib_ip.ip_preprocessing as pre
import lib_ip.ip_draw as draw
import lib_ip.ip_detection as det
import lib_ip.ip_segment as seg
import lib_ip.file_utils as file
import lib_ip.ocr_classify_text as ocr
import lib_ip.ip_detection_utils as util
import lib_ip.block_division as blk
import lib_ip.Component as Compo
from config.CONFIG_UIED import Config
C = Config()


def processing_block(org, binary, blocks, block_pad):
    # get binary map for each block
    blocks_clip_bin = seg.clipping(binary, blocks)
    image_shape = org.shape
    uicompos_all = []
    for i in range(len(blocks)):
        # *** Substep 1.1 *** pre-processing: get valid block -> binarization -> remove conglutinated line
        block = blocks[i]
        block_clip_bin = blocks_clip_bin[i]
        if block.block_is_top_or_bottom_bar(image_shape, C.THRESHOLD_TOP_BOTTOM_BAR):
            continue
        if block.block_is_uicompo(image_shape, C.THRESHOLD_COMPO_MAX_SCALE):
            uicompos_all.append(block)
            continue
        # det.line_removal(block_clip_bin, show=True)
        for i in block.children:
            blocks[i].block_erase_from_bin(binary, block_pad)

        # *** Substep 1.2 *** object extraction: extract components boundary -> get bounding box corner
        uicompos = det.component_detection(block_clip_bin)
        uicompos = Compo.cvt_compos_relative_pos(uicompos, block.bbox.col_min, block.bbox.row_min)
        uicompos_all += uicompos
    return uicompos_all


def compo_detection(input_img_path, output_root, num=0, resize_by_height=600, block_pad=4,
                    classifier=None, show=False, write_img=True):
    start = time.clock()
    name = input_img_path.split('\\')[-1][:-4]
    ip_root = file.build_directory(pjoin(output_root, "ip"))
    cls_root = file.build_directory(pjoin(output_root, "cls"))

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary = pre.preprocess(org, show=show, write_path=pjoin(ip_root, name + '_binary.png') if write_img else None)

    # *** Step 2 *** block processing: detect block -> detect components in block
    blocks = blk.block_division(grey, show=show, write_path=pjoin(ip_root, name + '_block.png') if write_img else None)
    blk.block_hierarchy(blocks)
    uicompos_in_blk = processing_block(org, binary, blocks, block_pad)

    # *** Step 3 *** non-block processing: erase blocks from binary -> detect left components
    det.rm_line(binary)
    blk.block_bin_erase_all_blk(binary, blocks, block_pad)
    uicompos_not_in_blk = det.component_detection(binary)
    uicompos = uicompos_in_blk + uicompos_not_in_blk

    # *** Step 4 *** results refinement: remove top and bottom compos -> merge words into line
    uicompos = det.rm_top_or_bottom_corners(uicompos, org.shape)
    file.save_corners_json(pjoin(ip_root, name + '_all.json'), uicompos + blocks)
    draw.draw_bounding_box(org, uicompos, show=show)

    # *** Step 5 *** post-processing: merge components -> classification (opt)
    if classifier is not None:
        classifier.predict(seg.clipping(org, uicompos), uicompos)
        draw.draw_bounding_box_class(org, uicompos, show=show, write_path=pjoin(cls_root, name + '.png'))
        file.save_corners_json(pjoin(cls_root, name + '.json'), uicompos)
    uicompos = det.merge_text(uicompos, org.shape)
    uicompos = det.merge_intersected_corner(uicompos, org.shape)

    # *** Step 6 *** save results: save text label -> save drawn image
    draw.draw_bounding_box(org, uicompos, show=show, write_path=pjoin(ip_root, name + '_ip.png'))
    file.save_corners_json(pjoin(ip_root, name + '_ip.json'), uicompos)

    print("[Compo Detection Completed in %.3f s] %d %s\n" % (time.clock() - start, num, input_img_path))
