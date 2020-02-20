import ip_region_proposal as ip
import time

resize_by_height = 800

# set input image path
PATH_IMG_INPUT = 'E:\\Mulong\\Datasets\\rico\\combined\\23.jpg'
PATH_OUTPUT_ROOT = 'data\\output'

is_clf = False
classifier = None
if is_clf:
    from Resnet import ResClassifier
    classifier = ResClassifier()

ip.compo_detection(PATH_IMG_INPUT, PATH_OUTPUT_ROOT,
                   resize_by_height=resize_by_height, classifier=classifier,
                   show=True, write_img=False)
