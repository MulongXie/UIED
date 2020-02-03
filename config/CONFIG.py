from os.path import join as pjoin
import os


class Config:

    def __init__(self):
        # setting CNN model
        self.image_shape = (64, 64, 3)
        self.class_map = ['button', 'input', 'icon', 'img', 'text']
        self.class_number = len(self.class_map)
        self.MODEL_PATH = 'E:/Mulong/Model/UI2CODE/cnn6_icon.h5'

        # setting CTPN (ocr) model
        self.CTPN_PATH = "E:/Mulong/Model/UI2CODE/ctpn.pb"

        # setting dataflow paths
        # used for continuously processing
        self.ROOT_INPUT = "E:/Mulong/Datasets/google_play/data/play_store_screenshots"
        self.ROOT_OUTPUT = "E:/Mulong/Result/manually_label"

        # *** Frozen ***
        self.ROOT_IMG_ORG = pjoin(self.ROOT_INPUT, "org")
        self.ROOT_LABEL_UIED = pjoin(self.ROOT_OUTPUT, "ui_label")
        self.ROOT_IMG_DRAWN_UIED = pjoin(self.ROOT_OUTPUT, "ui_img_drawn")
        self.ROOT_IMG_GRADIENT_UIED = pjoin(self.ROOT_OUTPUT, "ui_img_gradient")
        self.ROOT_LABEL_CTPN = pjoin(self.ROOT_OUTPUT, "ctpn_label")
        self.ROOT_IMG_DRAWN_CTPN = pjoin(self.ROOT_OUTPUT, "ctpn_drawn")
        self.ROOT_IMG_MERGE = pjoin(self.ROOT_OUTPUT, "merge_drawn")
        self.ROOT_IMG_COMPONENT = pjoin(self.ROOT_OUTPUT, "components")
        self.COLOR = {'block': (0, 255, 0), 'img': (0, 0, 255), 'icon': (255, 166, 166), 'input': (255, 166, 0),
                      'text': (77, 77, 255), 'search': (255, 0, 166), 'list': (166, 0, 255), 'select': (166, 166, 166),
                      'button': (0, 166, 255)}
        self.class_index = {'button':0, 'input':1, 'select':2, 'search':3, 'list':4, 'img':5, 'block':6, 'text':7, 'icon':8}

    def build_output_folders(self, is_clip):
        if not os.path.exists(self.ROOT_LABEL_UIED):
            os.mkdir(self.ROOT_LABEL_UIED)
        if not os.path.exists(self.ROOT_IMG_DRAWN_UIED):
            os.mkdir(self.ROOT_IMG_DRAWN_UIED)
        if not os.path.exists(self.ROOT_IMG_GRADIENT_UIED):
            os.mkdir(self.ROOT_IMG_GRADIENT_UIED)
        if not os.path.exists(self.ROOT_LABEL_CTPN):
            os.mkdir(self.ROOT_LABEL_CTPN)
        if not os.path.exists(self.ROOT_IMG_DRAWN_CTPN):
            os.mkdir(self.ROOT_IMG_DRAWN_CTPN)
        if not os.path.exists(self.ROOT_IMG_MERGE):
            os.mkdir(self.ROOT_IMG_MERGE)
        if is_clip and not os.path.exists(self.ROOT_IMG_COMPONENT):
            os.mkdir(self.ROOT_IMG_COMPONENT)
