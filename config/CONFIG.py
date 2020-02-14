from os.path import join as pjoin
import os


class Config:

    def __init__(self):
        # setting CNN model
        self.image_shape = (64, 64, 3)
        self.class_map = ['button', 'input', 'icon', 'img', 'text']
        self.class_number = len(self.class_map)
        self.MODEL_PATH = 'E:\\Mulong\\Model\\UI2CODE\\cnn6_icon.h5'

        # setting CTPN (ocr) model
        self.CTPN_PATH = "E:\\Mulong\\Model\\UI2CODE\\ctpn.pb"

        # setting data flow paths
        self.ROOT_INPUT = "E:\\Mulong\\Datasets\\rico\\combined"
        self.ROOT_OUTPUT = "E:\\Mulong\\Result\\rico2"

        # *** Frozen ***
        self.ROOT_IMG_ORG = pjoin(self.ROOT_INPUT, "org")
        self.ROOT_IP = pjoin(self.ROOT_OUTPUT, "ip")
        self.ROOT_OCR = pjoin(self.ROOT_OUTPUT, "ocr")
        self.ROOT_MERGE = pjoin(self.ROOT_OUTPUT, "merge")
        self.ROOT_IMG_COMPONENT = pjoin(self.ROOT_OUTPUT, "components")
        self.COLOR = {'block': (0, 255, 0), 'img': (0, 0, 255), 'icon': (255, 166, 166), 'input': (255, 166, 0),
                      'text': (77, 77, 255), 'search': (255, 0, 166), 'list': (166, 0, 255), 'select': (166, 166, 166),
                      'button': (0, 166, 255)}
        self.class_index = {'button':0, 'input':1, 'select':2, 'search':3, 'list':4, 'img':5, 'block':6, 'text':7, 'icon':8}

    def build_output_folders(self, is_clip):
        if not os.path.exists(self.ROOT_IP):
            os.mkdir(self.ROOT_IP)
        if not os.path.exists(self.ROOT_OCR):
            os.mkdir(self.ROOT_OCR)
        if not os.path.exists(self.ROOT_MERGE):
            os.mkdir(self.ROOT_MERGE)
        if is_clip and not os.path.exists(self.ROOT_IMG_COMPONENT):
            os.mkdir(self.ROOT_IMG_COMPONENT)
