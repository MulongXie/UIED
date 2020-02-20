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
from config.CONFIG_UIED import Config


def processing_block(org, binary, blocks_corner, block_pad):
    # get binary map for each block
    blocks_corner = det.corner_padding(org, blocks_corner, block_pad)
    blocks_clip_bin = seg.clipping(binary, blocks_corner)

    all_compos_corner = []
    for i in range(len(blocks_corner)):
        # *** Substep 1.1 *** pre-processing: get valid block -> binarization -> remove conglutinated line
        block_corner = blocks_corner[i]
        block_clip_bin = blocks_clip_bin[i]
        if det.is_top_or_bottom_bar(blocks_corner[i], org):
            continue
        if blk.block_is_compo(block_corner, org):
            all_compos_corner.append(block_corner)
            continue
        det.line_removal(block_clip_bin)

        # *** Substep 1.2 *** object extraction: extract components boundary -> get bounding box corner
        compos_boundary = det.boundary_detection(block_clip_bin)
        compos_corner = det.get_corner(compos_boundary)
        compos_corner = util.corner_cvt_relative_position(compos_corner, block_corner[0][0], block_corner[0][1])
        all_compos_corner += compos_corner
    return all_compos_corner


def processing(org, binary):
    # *** Substep 2.1 *** pre-processing: remove conglutinated line
    det.line_removal(binary, show=False)

    # *** Substep 2.2 *** object extraction: extract components boundary -> get bounding box corner
    compos_boundary = det.boundary_detection(binary)
    compos_corner = det.get_corner(compos_boundary)
    return compos_corner


def compo_detection(input_img_path, output_root, num=0, resize_by_height=600, block_pad=4,
                    classifier=None, show=False, write_img=True):
    start = time.clock()
    name = input_img_path.split('\\')[-1][:-4]
    ip_root = file.build_directory(pjoin(output_root, "ip"))
    cls_root = file.build_directory(pjoin(output_root, "cls"))

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary_org = pre.preprocess(org, write_path=pjoin(ip_root, name + '_binary.png') if write_img else None)

    # *** Step 2 *** block processing: detect block -> detect components in block
    blocks_corner = blk.block_division(grey, write_path=pjoin(ip_root, name + '_block.png') if write_img else None)
    compo_in_blk_corner = processing_block(org, binary_org, blocks_corner, block_pad)

    # *** Step 3 *** non-block processing: erase blocks from binary -> detect left components
    binary_non_block = blk.block_erase(binary_org, blocks_corner, pad=block_pad)
    compo_non_blk_corner = processing(org, binary_non_block)

    # *** Step 4 *** results refinement: remove top and bottom compos -> merge words into line
    compos_corner = compo_in_blk_corner + compo_non_blk_corner
    compos_corner = det.rm_top_or_bottom_corners(compos_corner, org.shape)
    file.save_corners_json(pjoin(ip_root, name + '_all.json'), compos_corner + blocks_corner,
                           list(np.full(len(compos_corner), 'compo')) + list(np.full(len(compos_corner), 'block')))
    draw.draw_bounding_box(org, compos_corner, show=True)

    # *** Step 5 *** post-processing: merge components -> classification (opt)
    compos_corner = det.merge_text(compos_corner, org.shape)
    compos_corner = det.merge_intersected_corner(compos_corner, org.shape)
    if classifier is not None:
        compos_class = classifier.predict(seg.clipping(org, compos_corner))
        draw.draw_bounding_box_class(org, compos_corner, compos_class, show=show, write_path=pjoin(cls_root, name + '.png'))
        file.save_corners_json(pjoin(cls_root, name + '.json'), compos_corner, compos_class)

    # *** Step 6 *** save results: save text label -> save drawn image
    draw.draw_bounding_box(org, compos_corner, show=show, write_path=pjoin(ip_root, name + '_ip.png'))
    file.save_corners_json(pjoin(ip_root, name + '_ip.json'), compos_corner, np.full(len(compos_corner), '0'))

    print("[Compo Detection Completed in %.3f s] %d %s\n" % (time.clock() - start, num, input_img_path))
