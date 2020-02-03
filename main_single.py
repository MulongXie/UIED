import time

is_ctpn = True
is_uied = True
is_merge = True
resize_by_height = 600

# set input image path
PATH_IMG_INPUT = 'data/input/1.jpg'
PATH_OUTPUT_ROOT = 'data/output/'

start = time.clock()
if is_ctpn:
    import ocr as ocr
    ocr.ctpn(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
if is_uied:
    import ip as ip
    ip.compo_detection(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
if is_merge:
    import merge
    merge.incorporate(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height)
print('Time Taken:%.3f s\n' % (time.clock() - start))
