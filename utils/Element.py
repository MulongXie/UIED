import numpy as np
from detect_compo.lib_ip.Bbox import Bbox


class Element:
    def __init__(self, corner, category):
        self.category = category
        self.bbox = Bbox(corner[0], corner[1], corner[2], corner[3])
        self.area = self.bbox.box_area
        self.width = self.bbox.width
        self.height = self.bbox.height

    def put_bbox(self):
        return self.bbox.put_bbox()

    def element_relation(self, element_b, bias=(0, 0)):
        """
        :return: -1 : a in b
                 0  : a, b are not intersected
                 1  : b in a
                 2  : a, b are identical or intersected
        """
        return self.bbox.bbox_relation_nms(element_b.bbox, bias)

    def element_merge(self, element_b, new_element=False, new_category=None):
        if not new_element:
            self.bbox = self.bbox.bbox_merge(element_b.bbox)
        else:
            bbox = self.bbox.bbox_merge(element_b.bbox)
            element = Element(bbox.put_bbox(), new_category)
            return element

    def calc_intersection_area(self, element_b):
        a = self.put_bbox()
        b = element_b.put_bbox()
        col_min_s = max(a[0], b[0])
        row_min_s = max(a[1], b[1])
        col_max_s = min(a[2], b[2])
        row_max_s = min(a[3], b[3])
        w = np.maximum(0, col_max_s - col_min_s)
        h = np.maximum(0, row_max_s - row_min_s)
        inter = w * h
        return inter

    def calc_iou(self, element_b):
        inter = self.calc_intersection_area(element_b)
        iou = inter / (self.area + element_b.area - inter)
        return iou

