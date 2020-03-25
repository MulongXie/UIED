from os.path import join as pjoin
import ip_region_proposal as ip

resize_by_height = 800

# set input image path
# PATH_IMG_INPUT = 'E:\\Mulong\\Datasets\\rico\\combined\\23.jpg'
input_path_img = 'data\\input\\a.png'
output_root = 'data\\output'

is_ip = True
is_clf = True
is_ocr = False
is_merge = True

if is_ocr:
    import ocr_east as ocr
    ocr.east(input_path_img, output_root, resize_by_height=None, show=False, write_img=True)

if is_ip:
    # switch of the classification func
    classifier = None
    if is_clf:
        classifier = {}
        from CNN import CNN

        classifier['Image'] = CNN('Image')
        classifier['Elements'] = CNN('Elements')
        classifier['Noise'] = CNN('Noise')

    ip.compo_detection(input_path_img, output_root, resize_by_height=resize_by_height, show=False,
                       classifier=classifier)

if is_merge:
    import merge
    name = input_path_img.split('\\')[-1][:-4]
    compo_path = pjoin(output_root, 'ip', str(name) + '.json')
    ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
    merge.incorporate(input_path_img, compo_path, ocr_path, output_root, resize_by_height=resize_by_height, show=True,
                      write_img=True)
