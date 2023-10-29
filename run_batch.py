import multiprocessing
import glob
import time
import json
from tqdm import tqdm
from os.path import join as pjoin, exists
import cv2
import time
import requests
import detect_compo.ip_region_proposal as ip


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    # initialization
    input_img_root = "/public/dataset/UEyes/UEyes_dataset/images/"
    output_root = "/public/dataset/UEyes/UEyes_dataset/anno_uied"
    data = json.load(open('/public/dataset/UEyes/UEyes_dataset/images.json', 'r'))

    # input_imgs = [pjoin(input_img_root, img['file_name'].split('/')[-1]) for img in data['images']]
    # input_imgs = sorted(input_imgs, key=lambda x: int(x.split('/')[-1][:-4]))  # sorted by index
    input_imgs = [img['file_name'] for img in data['images']]
    
    # key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
    #               'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-bar': True}
    key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}

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
    
    for input_img in tqdm(input_imgs):
        resized_height = resize_height_by_longest_edge(input_img)
        if is_ocr:
            for i in range(max_retries):
                try:
                    text.text_detection(input_img, output_root, show=False)
                    break
                except requests.exceptions.ProxyError:
                    if i < max_retries - 1:
                        print("Request failed due to proxy error. Retrying in {} seconds...".format(retry_wait_seconds))
                        time.sleep(retry_wait_seconds)
                    else:
                        print("Request failed after maximum retries. Exiting.")
                        raise

        if is_ip:
            ip.compo_detection(input_img, output_root, key_params,  classifier=compo_classifier, resize_by_height=resized_height, show=False)

        if is_merge:
            import detect_merge.merge as merge
            name = input_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            # merge.merge(input_img, compo_path, ocr_path, output_root, 
            #             is_remove_top=key_params['remove-top-bar'], 
            #             show=True)
            merge.merge(input_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                    is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False)
