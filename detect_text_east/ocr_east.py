import time
from os.path import join as pjoin
import lib_east.eval as eval

sess, f_score, f_geometry, input_images = eval.load()


def east(input_img_path, output_label_path, resize_by_height, show=False, write_img=True):
    start = time.clock()
    # print("OCR Starts for %s" %input_img_path)
    output_label_path = pjoin(output_label_path, 'ocr')
    eval.run(input_img_path, output_label_path, resize_by_height,
             sess, f_score, f_geometry, input_images, show=show, write_img=write_img)
    print("[OCR Completed in %.3f s] %s" % (time.clock() - start, input_img_path))