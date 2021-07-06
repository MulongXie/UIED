import cv2
from os.path import join as pjoin
import time
import numpy as np

from detect_compo.lib_ip.Component import Component
from config.CONFIG_UIED import Config
C = Config()


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


class Block(Component):
    def __init__(self, region, image_shape):
        super().__init__(region, image_shape)
        self.category = 'Block'
        self.parent = None
        self.children = []
        self.uicompo_ = None
        self.top_or_botm = None
        self.redundant = False

    def block_is_uicompo(self, image_shape, max_compo_scale):
        '''
        Check the if the block is a ui component according to its relative size
        '''
        row, column = image_shape[:2]
        # print(height, height / row, max_compo_scale[0], height / row > max_compo_scale[0])
        # draw.draw_bounding_box(org, [corner], show=True)
        # ignore atomic components
        if self.bbox.height / row > max_compo_scale[0] or self.bbox.width / column > max_compo_scale[1]:
            return False
        return True

    def block_is_top_or_bottom_bar(self, image_shape, top_bottom_height):
        '''
        Check if the block is top bar or bottom bar
        '''
        height, width = image_shape[:2]
        (column_min, row_min, column_max, row_max) = self.bbox.put_bbox()
        if column_min < 5 and row_min < 5 and \
                width - column_max < 5 and row_max < height * top_bottom_height[0]:
            self.uicompo_ = True
            return True
        if column_min < 5 and row_min > height * top_bottom_height[1] and \
                width - column_max < 5 and height - row_max < 5:
            self.uicompo_ = True
            return True
        return False

    def block_erase_from_bin(self, binary, pad):
        (column_min, row_min, column_max, row_max) = self.put_bbox()
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, binary.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, binary.shape[0])
        cv2.rectangle(binary, (column_min, row_min), (column_max, row_max), (0), -1)

