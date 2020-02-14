class Config:

    def __init__(self):
        # Adjustable
        self.THRESHOLD_PRE_GRADIENT = 4             # dribbble:4 rico:4 web:1
        self.THRESHOLD_OBJ_MIN_AREA = 60
        self.THRESHOLD_OBJ_MIN_PERIMETER = 0
        self.THRESHOLD_BLOCK_GRADIENT = 8

        # *** Frozen ***
        self.THRESHOLD_REC_MIN_EVENNESS = 0.7
        self.THRESHOLD_REC_MAX_DENT_RATIO = 0.25
        self.THRESHOLD_LINE_THICKNESS = 8
        self.THRESHOLD_LINE_MIN_LENGTH = 0.9
        self.THRESHOLD_COMPO_MAX_SCALE = (0.1, 0.95)  # (80/800, 422.5/450) maximum height and width ratio for a atomic compo (button)
        self.THRESHOLD_TEXT_MAX_WORD_GAP = 10
        self.THRESHOLD_TEXT_MAX_HEIGHT = 0.05  # 40/800 maximum height of text
        self.THRESHOLD_TOP_BOTTOM_BAR = (0.045, 0.94)  # (36/800, 752/800) height ratio of top and bottom bar
        self.THRESHOLD_BLOCK_MIN_HEIGHT = 0.03  # 24/800

        # obsolete
        self.THRESHOLD_BLOCK_MAX_BORDER_THICKNESS = 8
        self.THRESHOLD_BLOCK_MAX_CROSS_POINT = 0.1
        self.THRESHOLD_UICOMPO_MIN_W_H_RATIO = 0.4
        self.THRESHOLD_TEXT_MAX_WIDTH = 150
        self.THRESHOLD_LINE_MIN_LENGTH_H = 50
        self.THRESHOLD_LINE_MIN_LENGTH_V = 50
        self.OCR_PADDING = 5
        self.OCR_MIN_WORD_AREA = 0.45

        self.THRESHOLD_MIN_IOU = 0.1              # dribbble:0.003 rico:0.1 web:0.1
        self.THRESHOLD_BLOCK_MIN_EDGE_LENGTH = 210   # dribbble:68 rico:210 web:70
        self.THRESHOLD_UICOMPO_MAX_W_H_RATIO = 10   # dribbble:10 rico:10 web:22

        self.COLOR = {'block': (0, 255, 0), 'img': (0, 0, 255), 'icon': (255, 166, 166), 'input': (255, 166, 0),
                      'text': (77, 77, 255), 'search': (255, 0, 166), 'list': (166, 0, 255), 'select': (166, 166, 166),
                      'button': (0, 166, 255)}
