import cv2
import numpy as np
from config.CONFIG_UIED import Config
C = Config()


def read_img(path, resize_height=None):

    def resize_by_height(org):
        w_h_ratio = org.shape[1] / org.shape[0]
        resize_w = resize_height * w_h_ratio
        re = cv2.resize(org, (int(resize_w), int(resize_height)))
        return re

    try:
        img = cv2.imread(path)
        if resize_height is not None:
            img = resize_by_height(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    except Exception as e:
        print(e)
        print("*** Img Reading Failed ***\n")
        return None, None


def gray_to_gradient(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, column = img.shape[0], img.shape[1]
    img_f = np.copy(img)
    img_f = img_f.astype("float")

    gradient = np.zeros((row, column))
    for x in range(row - 1):
        for y in range(column - 1):
            gx = abs(img_f[x + 1, y] - img_f[x, y])
            gy = abs(img_f[x, y + 1] - img_f[x, y])
            gradient[x, y] = gx + gy
    gradient = gradient.astype("uint8")
    return gradient


def grad_to_binary(grad, min):
    rec, bin = cv2.threshold(grad, min, 255, cv2.THRESH_BINARY)
    return bin


def reverse_binary(bin):
    """
    Reverse the input binary image
    """
    r, bin = cv2.threshold(bin, 1, 255, cv2.THRESH_BINARY_INV)
    return bin


def preprocess(org, grad_min=C.THRESHOLD_MIN_GRADIENT, write_path=None):
    grey = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    grad = gray_to_gradient(grey)        # get RoI with high gradient
    binary = grad_to_binary(grad, grad_min)   # enhance the RoI
    # morph = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (1, 1))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (5, 5))   # remove noises
    # morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, (3, 3))
    if write_path is not None:
        cv2.imwrite(write_path, morph)
    return morph
