import time
import lib_east.eval as eval

sess, f_score, f_geometry, input_images = eval.load()


def east(input_img_path, output_label_path, resize_by_height):
    start = time.clock()
    print("OCR Starts for %s" %input_img_path)
    eval.run(input_img_path, output_label_path, resize_by_height,
             sess, f_score, f_geometry, input_images)
    print("[OCR Completed in %.3f s]" % (time.clock() - start))