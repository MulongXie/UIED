import os
import time
import json
import argparse
from tqdm import tqdm
from os.path import join as pjoin
import cv2
import requests
import detect_compo.ip_region_proposal as ip
from paddleocr import PaddleOCR


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    # initialization
    parser = argparse.ArgumentParser(description='UIED batch processing')
    parser.add_argument('--cat', default='web', type=str, help='category of images')
    args = parser.parse_args()
    input_img_root = os.path.join(
        "/public/dataset/UEyes/UEyes_dataset/images_per_cat/", args.cat
    )
    categroy_list = ['mobile', 'web', 'desktop', 'poster']
    assert args.cat in categroy_list, 'category should be in {}'.format(categroy_list)
    output_root = os.path.join(
        "/public/dataset/UEyes/UEyes_dataset/anno_uied", args.cat
    )
    data = json.load(open('/public/dataset/UEyes/UEyes_dataset/images.json', 'r'))

    # input_imgs = [pjoin(input_img_root, img['file_name'].split('/')[-1]) for img in data['images']]
    # input_imgs = sorted(input_imgs, key=lambda x: int(x.split('/')[-1][:-4]))  # sorted by index
    input_imgs = [
        img['file_name'] for img in data['images'] if img['Category'] == args.cat
    ]

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
    key_params_dic = {
        'mobile': {
            'min-grad': 4,
            'ffl-block': 5,
            'min-ele-area': 50,
            'max-word-inline-gap': 6,
            'max-line-gap': 1,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': False,
            'remove-bar': False,
        },
        'web': {
            'min-grad': 3,
            'ffl-block': 5,
            'min-ele-area': 25,
            'max-word-inline-gap': 4,
            'max-line-gap': 4,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': False,
            'remove-bar': False,
        },
        'poster': {
            'min-grad': 5,
            'ffl-block': 3,
            'min-ele-area': 75,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': False,
            'remove-bar': False,
        },
        'desktop': {
            'min-grad': 5,
            'ffl-block': 5,
            'min-ele-area': 50,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': False,
            'remove-bar': True,
        },
    }
    key_params = key_params_dic[args.cat]
    is_ip = True
    is_clf = True
    is_ocr = False
    is_merge = True
    # Load deep learning models in advance
    compo_classifier = None
    if is_ip and is_clf:
        compo_classifier = {}
        from cnn.CNN import CNN

        # compo_classifier['Image'] = CNN('Image')
        compo_classifier['Elements'] = CNN('Elements')
        # compo_classifier['Noise'] = CNN('Noise')
    ocr_model = None
    if is_ocr:
        import detect_text.text_detection as text

    # set the range of target inputs' indices
    # num = 0
    # start_index = 30800  # 61728
    # end_index = 100000
    # for input_img in input_imgs:
    #     resized_height = resize_height_by_longest_edge(input_img)
    #     index = input_img.split('/')[-1][:-4]
    #     if int(index) < start_index:
    #         continue
    #     if int(index) > end_index:
    #         break
    max_retries = 5
    retry_wait_seconds = 5
    paddle_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    for input_img in tqdm(input_imgs):
        resized_height = resize_height_by_longest_edge(input_img)
        if is_ocr:
            for i in range(max_retries):
                try:
                    text.text_detection(
                        input_img,
                        output_root,
                        show=False,
                        method='paddle',
                        paddle_model=paddle_model,
                    )
                    break
                except requests.exceptions.ProxyError:
                    if i < max_retries - 1:
                        print(
                            "Request failed due to proxy error. Retrying in {} seconds...".format(
                                retry_wait_seconds
                            )
                        )
                        time.sleep(retry_wait_seconds)
                    else:
                        print("Request failed after maximum retries. Exiting.")
                        raise

        if is_ip:
            ip.compo_detection(
                input_img,
                output_root,
                key_params,
                classifier=compo_classifier,
                resize_by_height=resized_height,
                show=False,
            )

        if is_merge:
            import detect_merge.merge as merge

            name = input_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge_path = pjoin(output_root, 'merge')
            os.makedirs(merge_path, exist_ok=True)
            # merge.merge(input_img, compo_path, ocr_path, output_root,
            #             is_remove_top=key_params['remove-top-bar'],
            #             show=True)
            merge.merge(
                input_img,
                compo_path,
                ocr_path,
                merge_path,
                is_remove_bar=key_params['remove-bar'],
                is_paragraph=key_params['merge-line-to-paragraph'],
                show=False,
            )
