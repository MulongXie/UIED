import time
import lib_east.eval


def east(input_img_path, output_label_path, resize_by_height):
    start = time.clock()
    print("*** OCR Starts for %s ***" %input_img_path)
    lib_east.eval.run(input_img_path, output_label_path, resize_by_height)
    print("*** OCR Completed in %.3f s ***" % (time.clock() - start))