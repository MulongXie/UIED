import detect_text.ocr as ocr
from detect_text.Text import Text
import cv2
import json
import os


def save_detection_json(file_path, texts, img_shape):
    f_out = open(file_path, 'w')
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = loc['left'], loc['top'], loc['right'], loc['bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)

    json.dump(output, f_out, indent=4)


def show_texts(org_img, texts, shown_resize=None):
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)
    if shown_resize is not None:
        img = cv2.resize(img, shown_resize)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def text_sentences_recognition(texts, bias_justify, bias_gap):
    '''
    Merge separate words detected by Google ocr into a sentence
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(text_b, 'h', bias_justify=bias_justify, bias_gap=bias_gap):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def text_cvt_orc_format(ocr_result):
    texts = []
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            x_coordinates = []
            y_coordinates = []
            text_location = result['boundingPoly']['vertices']
            content = result['description']
            for loc in text_location:
                x_coordinates.append(loc['x'])
                y_coordinates.append(loc['y'])
            location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                        'right': max(x_coordinates), 'bottom': max(y_coordinates)}
            texts.append(Text(i, content, location))
    return texts


def text_detection(input_file='../data/input/30800.jpg', output_file='../data/output/ocr/30800'):
    img = cv2.imread(input_file)
    ocr_result = ocr.ocr_detection_google(input_file)
    texts = text_cvt_orc_format(ocr_result)
    show_texts(img, texts, (600, 900))
    texts = text_sentences_recognition(texts, bias_justify=5, bias_gap=50)
    show_texts(img, texts, (600, 900))
    save_detection_json(output_file + '.json', texts, img.shape)


text_detection()

