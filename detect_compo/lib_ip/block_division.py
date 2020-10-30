import cv2
import numpy as np
from random import randint as rint
import time

import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_segment as seg
from detect_compo.lib_ip.Block import Block
from config.CONFIG_UIED import Config
C = Config()


def block_hierarchy(blocks):
    for i in range(len(blocks) - 1):
        for j in range(i + 1, len(blocks)):
            relation = blocks[i].compo_relation(blocks[j])
            if relation == -1:
                blocks[j].children.append(i)
            if relation == 1:
                blocks[i].children.append(j)
    return


def block_bin_erase_all_blk(binary, blocks, pad=0, show=False):
    '''
    erase the block parts from the binary map
    :param binary: binary map of original image
    :param blocks_corner: corners of detected layout block
    :param show: show or not
    :param pad: expand the bounding boxes of blocks
    :return: binary map without block parts
    '''

    bin_org = binary.copy()
    for block in blocks:
        block.block_erase_from_bin(binary, pad)
    if show:
        cv2.imshow('before', bin_org)
        cv2.imshow('after', binary)
        cv2.waitKey()


def block_division(grey, org, grad_thresh,
                   show=False, write_path=None,
                   step_h=10, step_v=10,
                   line_thickness=C.THRESHOLD_LINE_THICKNESS,
                   min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                   max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                   min_block_height_ratio=C.THRESHOLD_BLOCK_MIN_HEIGHT):
    '''
    :param grey: grey-scale of original image
    :return: corners: list of [(top_left, bottom_right)]
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    '''
    blocks = []
    mask = np.zeros((grey.shape[0]+2, grey.shape[1]+2), dtype=np.uint8)
    broad = np.zeros((grey.shape[0], grey.shape[1], 3), dtype=np.uint8)
    broad_all = broad.copy()

    row, column = grey.shape[0], grey.shape[1]
    for x in range(0, row, step_h):
        for y in range(0, column, step_v):
            if mask[x, y] == 0:
                # region = flood_fill_bfs(grey, x, y, mask)

                # flood fill algorithm to get background (layout block)
                mask_copy = mask.copy()
                ff = cv2.floodFill(grey, mask, (y, x), None, grad_thresh, grad_thresh, cv2.FLOODFILL_MASK_ONLY)
                # ignore small regions
                if ff[0] < 500: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                block = Block(region, grey.shape)
                # draw.draw_region(region, broad_all)
                # if block.height < 40 and block.width < 40:
                #     continue
                if block.height < 30:
                    continue

                # print(block.area / (row * column))
                if block.area / (row * column) > 0.9:
                    continue
                elif block.area / (row * column) > 0.7:
                    block.redundant = True

                # get the boundary of this region
                # ignore lines
                if block.compo_is_line(line_thickness):
                    continue
                # ignore non-rectangle as blocks must be rectangular
                if not block.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                    continue
                # if block.height/row < min_block_height_ratio:
                #     continue
                blocks.append(block)
                # draw.draw_region(region, broad)
    if show:
        cv2.imshow('flood-fill all', broad_all)
        cv2.imshow('block', broad)
        cv2.waitKey()
    if write_path is not None:
        cv2.imwrite(write_path, broad)
    return blocks
