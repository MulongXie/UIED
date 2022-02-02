import cv2
import os
import io
from google.cloud import vision
import json
from base64 import b64encode
import time


def ocr_detection_google(imgpath):
    start = time.clock()
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
        KEY_PATH = "" # Set the path to the private key file created in https://cloud.google.com/vision/docs/setup#sa-create
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH
    try:
        client = vision.ImageAnnotatorClient()
    except Exception as e:
        print("*** Please export GOOGLE_APPLICATION_CREDENTIALS to the environment (apply in https://cloud.google.com/vision) ***")
        print(f"Exception {e}")
        return None
    with io.open(imgpath, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    print('*** Text Detection Time Taken:%.3fs ***' % (time.clock() - start))

    if not response.text_annotations:
        # No Text
        return None
    else:
        return response.text_annotations[1:]
