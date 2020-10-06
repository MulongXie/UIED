import time
from os.path import join as pjoin
import detect_text_east.lib_east.eval as eval
import os


def east(input_img_path, output_label_path, models, max_word_gap,
         resize_by_height=None, show=False, write_img=True):
    start = time.clock()
    sess, f_score, f_geometry, input_images = models
    # print("OCR Starts for %s" %input_img_path)
    output_label_path = pjoin(output_label_path, 'ocr')
    eval.run(input_img_path, output_label_path, max_word_gap, resize_by_height,
             sess, f_score, f_geometry, input_images, show=show, write_img=write_img)
    print("[OCR Completed in %.3f s]" % (time.clock() - start))
