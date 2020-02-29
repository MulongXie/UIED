
class Config:
    def __init__(self):
        # cnn 4 classes
        # self.MODEL_PATH = 'E:/Mulong/Model/ui_compos/cnn6_icon.h5'   # cnn 4 classes
        # self.class_map = ['Image', 'Icon', 'Button', 'Input']

        # resnet 14 classes
        # self.DATA_PATH = "E:/Mulong/Datasets/rico/elements-14-2"
        # self.MODEL_PATH = 'E:/Mulong/Model/rico_compos/resnet-ele14.h5'
        # self.class_map = ['Button', 'CheckBox', 'Chronometer', 'EditText', 'ImageButton', 'ImageView',
        #                   'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar', 'Spinner', 'Switch',
        #                   'ToggleButton', 'VideoView', 'TextView']  # ele-14

        self.DATA_PATH = "E:\Mulong\Datasets\dataset_webpage\Components3"

        self.MODEL_PATH = 'E:/Mulong/Model/rico_compos/cnn2-textview.h5'
        self.class_map = ['Text', 'Non-Text']

        self.image_shape = (32, 32, 3)
        self.class_number = len(self.class_map)
