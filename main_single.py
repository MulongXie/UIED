import ip_region_proposal as ip
import time

resize_by_height = 800

# set input image path
PATH_IMG_INPUT = 'data\\input\\x.jpg'
PATH_OUTPUT_ROOT = 'data\\output'

is_clf = True
classifier = None
if is_clf:
    from Resnet import ResClassifier
    classifier = ResClassifier()

ip.compo_detection(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height=resize_by_height, classifier=classifier, show=False)
