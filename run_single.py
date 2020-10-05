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


if __name__ == '__main__':

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffd-block: fill-flood threshold to segment layout block
        ele:min-ele-area: minimum area for selected elements 
        text:max-word-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph
    '''
    key_params = {'min-grad':5, 'ffd-block':5, 'min-ele-area':25,
                  'max-word-gap':4, 'max-line-gap':4}

    # set input image path
    input_path_img = 'data/input/9.png'
    output_root = 'data/output'

    resized_height = resize_height_by_longest_edge(input_path_img)

    is_ip = True
    is_clf = False
    is_ocr = False
    is_merge = True

    if is_ocr:
        import detect_text_east.ocr_east as ocr
        import detect_text_east.lib_east.eval as eval
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        models = eval.load()
        ocr.east(input_path_img, output_root, models, key_params['max-word-gap'],
                 resize_by_height=resized_height, show=False)

    if is_ip:
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
                           classifier=classifier, resize_by_height=resized_height, show=True)

    if is_merge:
        import merge
        name = input_path_img.split('/')[-1][:-4]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        merge.incorporate(input_path_img, compo_path, ocr_path, output_root, params=key_params,
                          resize_by_height=resized_height, show=True)
