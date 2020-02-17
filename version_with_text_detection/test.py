import ocr_east as ocr

PATH_IMG_INPUT = 'E:\\Mulong\\Datasets\\rico\\combined\\866.jpg'
PATH_OUTPUT_ROOT = 'detect_text_east\\lib_east\\data\\output'

ocr.east(PATH_IMG_INPUT, PATH_OUTPUT_ROOT, resize_by_height=None, show=True)