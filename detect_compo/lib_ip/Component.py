from detect_compo.lib_ip.Bbox import Bbox
import detect_compo.lib_ip.ip_draw as draw

import cv2


def cvt_compos_relative_pos(compos, col_min_base, row_min_base):
    for compo in compos:
        compo.compo_relative_position(col_min_base, row_min_base)


def compos_containment(compos):
    for i in range(len(compos) - 1):
        for j in range(i + 1, len(compos)):
            relation = compos[i].compo_relation(compos[j])
            if relation == -1:
                compos[j].contain.append(i)
            if relation == 1:
                compos[i].contain.append(j)


def compos_update(compos, org_shape):
    for i, compo in enumerate(compos):
        # start from 1, id 0 is background
        compo.compo_update(i + 1, org_shape)


class Component:
    def __init__(self, region, image_shape):
        self.id = None
        self.region = region
        self.boundary = self.compo_get_boundary()
        self.bbox = self.compo_get_bbox()
        self.bbox_area = self.bbox.box_area

        self.region_area = len(region)
        self.width = len(self.boundary[0])
        self.height = len(self.boundary[2])
        self.image_shape = image_shape
        self.area = self.width * self.height

        self.category = 'Compo'
        self.contain = []

        self.rect_ = None
        self.line_ = None
        self.redundant = False

    def compo_update(self, id, org_shape):
        self.id = id
        self.image_shape = org_shape
        self.width = self.bbox.width
        self.height = self.bbox.height
        self.bbox_area = self.bbox.box_area
        self.area = self.width * self.height

    def put_bbox(self):
        return self.bbox.put_bbox()

    def compo_update_bbox_area(self):
        self.bbox_area = self.bbox.bbox_cal_area()

    def compo_get_boundary(self):
        '''
        get the bounding boundary of an object(region)
        boundary: [top, bottom, left, right]
        -> up, bottom: (column_index, min/max row border)
        -> left, right: (row_index, min/max column border) detect range of each row
        '''
        border_up, border_bottom, border_left, border_right = {}, {}, {}, {}
        for point in self.region:
            # point: (row_index, column_index)
            # up, bottom: (column_index, min/max row border) detect range of each column
            if point[1] not in border_up or border_up[point[1]] > point[0]:
                border_up[point[1]] = point[0]
            if point[1] not in border_bottom or border_bottom[point[1]] < point[0]:
                border_bottom[point[1]] = point[0]
            # left, right: (row_index, min/max column border) detect range of each row
            if point[0] not in border_left or border_left[point[0]] > point[1]:
                border_left[point[0]] = point[1]
            if point[0] not in border_right or border_right[point[0]] < point[1]:
                border_right[point[0]] = point[1]

        boundary = [border_up, border_bottom, border_left, border_right]
        # descending sort
        for i in range(len(boundary)):
            boundary[i] = [[k, boundary[i][k]] for k in boundary[i].keys()]
            boundary[i] = sorted(boundary[i], key=lambda x: x[0])
        return boundary

    def compo_get_bbox(self):
        """
        Get the top left and bottom right points of boundary
        :param boundaries: boundary: [top, bottom, left, right]
                            -> up, bottom: (column_index, min/max row border)
                            -> left, right: (row_index, min/max column border) detect range of each row
        :return: corners: [(top_left, bottom_right)]
                            -> top_left: (column_min, row_min)
                            -> bottom_right: (column_max, row_max)
        """
        col_min, row_min = (int(min(self.boundary[0][0][0], self.boundary[1][-1][0])), int(min(self.boundary[2][0][0], self.boundary[3][-1][0])))
        col_max, row_max = (int(max(self.boundary[0][0][0], self.boundary[1][-1][0])), int(max(self.boundary[2][0][0], self.boundary[3][-1][0])))
        bbox = Bbox(col_min, row_min, col_max, row_max)
        return bbox

    def compo_is_rectangle(self, min_rec_evenness, max_dent_ratio, test=False):
        '''
        detect if an object is rectangle by evenness and dent of each border
        '''
        dent_direction = [1, -1, 1, -1]  # direction for convex

        flat = 0
        parameter = 0
        for n, border in enumerate(self.boundary):
            parameter += len(border)
            # dent detection
            pit = 0  # length of pit
            depth = 0  # the degree of surface changing
            if n <= 1:
                adj_side = max(len(self.boundary[2]), len(self.boundary[3]))  # get maximum length of adjacent side
            else:
                adj_side = max(len(self.boundary[0]), len(self.boundary[1]))

            # -> up, bottom: (column_index, min/max row border)
            # -> left, right: (row_index, min/max column border) detect range of each row
            abnm = 0
            for i in range(int(3 + len(border) * 0.02), len(border) - 1):
                # calculate gradient
                difference = border[i][1] - border[i + 1][1]
                # the degree of surface changing
                depth += difference
                # ignore noise at the start of each direction
                if i / len(border) < 0.08 and (dent_direction[n] * difference) / adj_side > 0.5:
                    depth = 0  # reset

                # print(border[i][1], i / len(border), depth, (dent_direction[n] * difference) / adj_side)
                # if the change of the surface is too large, count it as part of abnormal change
                if abs(depth) / adj_side > 0.3:
                    abnm += 1  # count the size of the abnm
                    # if the abnm is too big, the shape should not be a rectangle
                    if abnm / len(border) > 0.1:
                        if test:
                            print('abnms', abnm, abnm / len(border))
                            draw.draw_boundary([self], self.image_shape, show=True)
                        self.rect_ = False
                        return False
                    continue
                else:
                    # reset the abnm if the depth back to normal
                    abnm = 0

                # if sunken and the surface changing is large, then counted as pit
                if dent_direction[n] * depth < 0 and abs(depth) / adj_side > 0.15:
                    pit += 1
                    continue

                # if the surface is not changing to a pit and the gradient is zero, then count it as flat
                if abs(depth) < 1 + adj_side * 0.015:
                    flat += 1
                if test:
                    print(depth, adj_side, flat)
            # if the pit is too big, the shape should not be a rectangle
            if pit / len(border) > max_dent_ratio:
                if test:
                    print('pit', pit, pit / len(border))
                    draw.draw_boundary([self], self.image_shape, show=True)
                self.rect_ = False
                return False
        if test:
            print(flat / parameter, '\n')
            draw.draw_boundary([self], self.image_shape, show=True)
        # ignore text and irregular shape
        if self.height / self.image_shape[0] > 0.3:
            min_rec_evenness = 0.85
        if (flat / parameter) < min_rec_evenness:
            self.rect_ = False
            return False
        self.rect_ = True
        return True

    def compo_is_line(self, min_line_thickness):
        """
        Check this object is line by checking its boundary
        :param boundary: boundary: [border_top, border_bottom, border_left, border_right]
                                    -> top, bottom: list of (column_index, min/max row border)
                                    -> left, right: list of (row_index, min/max column border) detect range of each row
        :param min_line_thickness:
        :return: Boolean
        """
        # horizontally
        slim = 0
        for i in range(self.width):
            if abs(self.boundary[1][i][1] - self.boundary[0][i][1]) <= min_line_thickness:
                slim += 1
        if slim / len(self.boundary[0]) > 0.93:
            self.line_ = True
            return True
        # vertically
        slim = 0
        for i in range(self.height):
            if abs(self.boundary[2][i][1] - self.boundary[3][i][1]) <= min_line_thickness:
                slim += 1
        if slim / len(self.boundary[2]) > 0.93:
            self.line_ = True
            return True
        self.line_ = False
        return False

    def compo_relation(self, compo_b, bias=(0, 0)):
        """
        :return: -1 : a in b
                 0  : a, b are not intersected
                 1  : b in a
                 2  : a, b are identical or intersected
        """
        return self.bbox.bbox_relation_nms(compo_b.bbox, bias)

    def compo_relative_position(self, col_min_base, row_min_base):
        '''
        Convert to relative position based on base coordinator
        '''
        self.bbox.bbox_cvt_relative_position(col_min_base, row_min_base)

    def compo_merge(self, compo_b):
        self.bbox = self.bbox.bbox_merge(compo_b.bbox)
        self.compo_update(self.id, self.image_shape)

    def compo_clipping(self, img, pad=0, show=False):
        (column_min, row_min, column_max, row_max) = self.put_bbox()
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, img.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, img.shape[0])
        clip = img[row_min:row_max, column_min:column_max]
        if show:
            cv2.imshow('clipping', clip)
            cv2.waitKey()
        return clip
