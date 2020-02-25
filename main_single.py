import ip_region_proposal as ip
import time

resize_by_height = 800

# set input image path
PATH_IMG_INPUT = 'E:\\Mulong\\Datasets\\rico\\combined\\17.jpg'
PATH_OUTPUT_ROOT = 'data\\output'

is_ip = True
is_clf = False
is_ocr = False
is_merge = False

if is_ocr:
    import ocr_east as ocr
    ocr.east(PATH_IMG_INPUT, PATH_OUTPUT_ROOT,
             resize_by_height=resize_by_height, show=True, write_img=True)

if is_ip:
    # Turn on classification
    if is_clf:
        from Resnet import ResClassifier
        classifier = ResClassifier()
    else:
        classifier = None
    ip.compo_detection(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, classifier=classifier,
                       resize_by_height=resize_by_height, show=True, write_img=True)

if is_merge:
    import merge
    merge.incorporate(PATH_IMG_INPUT, PATH_OUTPUT_ROOT,
                      resize_by_height=resize_by_height, show=True, write_img=True)