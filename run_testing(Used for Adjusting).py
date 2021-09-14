from os.path import join as pjoin
import cv2
import os


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def nothing(x):
    pass


if __name__ == '__main__':

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50, 'merge-contained-ele':False,
                  'max-word-inline-gap':10, 'max-line-gap':4, 'remove-top-bar':True}

    # set input image path
    input_path_img = 'data/input/4.jpg'
    output_root = 'data/output'

    resized_height = resize_height_by_longest_edge(input_path_img)
    is_clf = False
    is_ocr = False
    if is_ocr:
        import detect_text.text_detection as text
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        text.text_detection(input_path_img, output_root, show=False)

    '''
    ******** Testing with adjustable parameters ********
    '''
    testing_ip = True
    testing_merge = False

    cv2.namedWindow('parameters')
    if testing_ip:
        cv2.createTrackbar('min-grad', 'parameters', 4, 20, nothing)
        cv2.createTrackbar('min-ele-area', 'parameters', 20, 200, nothing)
        while(1):
            key_params['min-grad'] = cv2.getTrackbarPos('min-grad', 'parameters')
            key_params['min-ele-area'] = cv2.getTrackbarPos('min-ele-area', 'parameters')
            import detect_compo.ip_region_proposal as ip
            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            # switch of the classification func
            classifier = None
            if is_clf:
                classifier = {}
                from cnn.CNN import CNN
                # classifier['Image'] = CNN('Image')
                classifier['Elements'] = CNN('Elements')
                # classifier['Noise'] = CNN('Noise')
            ip.compo_detection(input_path_img, output_root, key_params,
                               classifier=classifier, resize_by_height=resized_height, show=True, wai_key=10)

    if testing_merge:
        cv2.createTrackbar('max-word-inline-gap', 'parameters', 4, 20, nothing)
        cv2.createTrackbar('max-line-gap', 'parameters', 20, 200, nothing)
        while(1):
            key_params['max-word-inline-gap'] = cv2.getTrackbarPos('max-word-inline-gap', 'parameters')
            key_params['max-line-gap'] = cv2.getTrackbarPos('max-line-gap', 'parameters')
            import detect_merge.merge as merge
            name = input_path_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.merge(input_path_img, compo_path, ocr_path, output_root=None, is_remove_top=key_params['remove-top-bar'], show=True, wait_key=10)
