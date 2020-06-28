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
    image_shape = org.shape
    uicompos_all = []
    for block in blocks:
        # *** Step 2.1 *** check: examine if the block is valid layout block
        if block.block_is_top_or_bottom_bar(image_shape, C.THRESHOLD_TOP_BOTTOM_BAR):
            continue
        if block.block_is_uicompo(image_shape, C.THRESHOLD_COMPO_MAX_SCALE):
            uicompos_all.append(block)

        # *** Step 2.2 *** binary map processing: erase children block -> clipping -> remove lines(opt)
        binary_copy = binary.copy()
        for i in block.children:
            blocks[i].block_erase_from_bin(binary_copy, block_pad)
        block_clip_bin = block.compo_clipping(binary_copy)
        # det.line_removal(block_clip_bin, show=True)

        # *** Step 2.3 *** component extraction: detect components in block binmap -> convert position to relative
        uicompos = det.component_detection(block_clip_bin)
        Compo.cvt_compos_relative_pos(uicompos, block.bbox.col_min, block.bbox.row_min)
        uicompos_all += uicompos
    return uicompos_all


def compo_detection(input_img_path, output_root,
                    num=0, resize_by_height=600, block_pad=4,
                    classifier=None, show=False, write_img=True):
    start = time.clock()
    name = input_img_path.split('\\')[-1][:-4]
    ip_root = file.build_directory(pjoin(output_root, "ip"))

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary = pre.binarization(org, show=show, write_path=pjoin(ip_root, name + '_binary.png') if write_img else None)
    binary_org = binary.copy()

    # *** Step 2 *** block processing: detect block -> calculate hierarchy -> detect components in block
    blocks = blk.block_division(grey, org, show=show, write_path=pjoin(ip_root, name + '_block.png') if write_img else None)
    blk.block_hierarchy(blocks)
    uicompos_in_blk = processing_block(org, binary, blocks, block_pad)

    # *** Step 3 *** non-block part processing: remove lines -> erase blocks from binary -> detect left components
    det.rm_line(binary, show=True)
    blk.block_bin_erase_all_blk(binary, blocks, block_pad)
    uicompos_not_in_blk = det.component_detection(binary)
    uicompos = uicompos_in_blk + uicompos_not_in_blk

    # *** Step 4 *** results refinement: remove top and bottom compos -> merge words into line
    uicompos = det.rm_top_or_bottom_corners(uicompos, org.shape)
    file.save_corners_json(pjoin(ip_root, name + '_all.json'), uicompos)
    uicompos = det.merge_text(uicompos, org.shape)
    draw.draw_bounding_box(org, uicompos, show=show)
    # uicompos = det.merge_intersected_corner(uicompos, org.shape)
    Compo.compos_containment(uicompos)
    # draw.draw_bounding_box(org, uicompos, show=show, write_path=pjoin(ip_root, name + '_ip.png') if write_img else None)

    # *** Step 5 *** Image Inspection: recognize image -> remove noise in image -> binarize with larger threshold and reverse -> rectangular compo detection
    if classifier is not None:
        classifier['Image'].predict(seg.clipping(org, uicompos), uicompos)
        draw.draw_bounding_box_class(org, uicompos, show=show)
        uicompos = det.rm_noise_in_large_img(uicompos, org)
        draw.draw_bounding_box_class(org, uicompos, show=show)
        det.detect_compos_in_img(uicompos, binary_org, org)
        draw.draw_bounding_box(org, uicompos, show=show)

    # if classifier is not None:
    #     classifier['Noise'].predict(seg.clipping(org, uicompos), uicompos)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     uicompos = det.rm_noise_compos(uicompos)

    # *** Step 6 *** element classification: all category classification
    if classifier is not None:
        classifier['Elements'].predict(seg.clipping(org, uicompos), uicompos)
        draw.draw_bounding_box_class(org, uicompos, show=show, write_path=pjoin(ip_root, name + '_cls.png'))

    uicompos = det.compo_filter(uicompos, org)
    draw.draw_bounding_box(org, uicompos, show=show)
    file.save_corners_json(pjoin(ip_root, name + '.json'), uicompos)

    print("[Compo Detection Completed in %.3f s] %d %s" % (time.clock() - start, num, input_img_path))
    # Record run time
    open('time.txt', 'a').write(str(round(time.clock() - start, 3)) + '\n')
    if show:
        cv2.destroyAllWindows()