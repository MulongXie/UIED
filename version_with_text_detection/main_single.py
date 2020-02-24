import time

is_ocr = False
is_ip = True
is_merge = False
resize_by_height = 800

# set input image path
PATH_IMG_INPUT = 'data\\input\\1.jpg'
PATH_OUTPUT_ROOT = 'data\\output'

start = time.clock()
if is_ocr:
    # import ocr_ctpn as ocr
    # ocr.ctpn(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
    import ocr_east as ocr
    ocr.east(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
if is_ip:
    import ip
    ip.compo_detection(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
if is_merge:
    import merge
    merge.incorporate(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
print('Time Taken:%.3f s\n' % (time.clock() - start))
