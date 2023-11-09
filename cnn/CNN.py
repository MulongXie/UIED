import keras
# from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

from config.CONFIG import Config
cfg = Config()


class CNN:
    def __init__(self, classifier_type, is_load=True):
        '''
        :param classifier_type: 'Text' or 'Noise' or 'Elements'
        '''
        self.data = None
        self.model = None

        self.classifier_type = classifier_type

        self.image_shape = (32,32,3)
        self.class_number = None
        self.class_map = None
        self.model_path = None
        self.classifier_type = classifier_type
        if is_load:
            self.load(classifier_type)

    def build_model(self, epoch_num, is_compile=True):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.image_shape)
        for layer in base_model.layers:
            layer.trainable = False
        self.model = Flatten()(base_model.output)
        self.model = Dense(128, activation='relu')(self.model)
        self.model = Dropout(0.5)(self.model)
        self.model = Dense(15, activation='softmax')(self.model)

        self.model = Model(inputs=base_model.input, outputs=self.model)
        if is_compile:
            self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            self.model.fit(self.data.X_train, self.data.Y_train, batch_size=64, epochs=epoch_num, verbose=1,
                           validation_data=(self.data.X_test, self.data.Y_test))

    def train(self, data, epoch_num=30):
        self.data = data
        self.build_model(epoch_num)
        self.model.save(self.model_path)
        print("Trained model is saved to", self.model_path)

    def load(self, classifier_type):
        if classifier_type == 'Text':
            self.model_path = 'E:/Mulong/Model/rico_compos/cnn-textview-2.h5'
            self.class_map = ['Text', 'Non-Text']
        elif classifier_type == 'Noise':
            self.model_path = 'E:/Mulong/Model/rico_compos/cnn-noise-1.h5'
            self.class_map = ['Noise', 'Non-Noise']
        elif classifier_type == 'Elements':
            # self.model_path = 'E:/Mulong/Model/rico_compos/resnet-ele14-19.h5'
            # self.model_path = 'E:/Mulong/Model/rico_compos/resnet-ele14-28.h5'
            # self.model_path = 'E:/Mulong/Model/rico_compos/resnet-ele14-45.h5'
            self.model_path = cfg.CNN_PATH
            self.class_map = cfg.element_class
            self.image_shape = (64, 64, 3)
        elif classifier_type == 'Image':
            self.model_path = 'E:/Mulong/Model/rico_compos/cnn-image-1.h5'
            self.class_map = ['Image', 'Non-Image']
        self.class_number = len(self.class_map)
        self.model = load_model(self.model_path)
        print('Model Loaded From', self.model_path)

    def preprocess_img(self, image):
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255).astype('float32')
        x = np.array([x])
        return x

    def predict(self, imgs, compos, load=False, show=False):
        """
        :type img_path: list of img path
        """
        if load:
            self.load(self.classifier_type)
        if self.model is None:
            print("*** No model loaded ***")
            return
        for i in range(len(imgs)):
            X = self.preprocess_img(imgs[i])
            # verbose=0: for no log output for keras model.
            Y = self.class_map[np.argmax(self.model.predict(X,verbose=0))]
            compos[i].category = Y
            if show:
                print(Y)
                cv2.imshow('element', imgs[i])
                cv2.waitKey()

    def evaluate(self, data, load=True):
        if load:
            self.load(self.classifier_type)
        X_test = data.X_test
        Y_test = [np.argmax(y) for y in data.Y_test]
        Y_pre = [np.argmax(y_pre) for y_pre in self.model.predict(X_test, verbose=1)]

        matrix = confusion_matrix(Y_test, Y_pre)
        print(matrix)

        TP, FP, FN = 0, 0, 0
        for i in range(len(matrix)):
            TP += matrix[i][i]
            FP += sum(matrix[i][:]) - matrix[i][i]
            FN += sum(matrix[:][i]) - matrix[i][i]
        precision = TP/(TP+FP)
        recall = TP / (TP+FN)
        print("Precision:%.3f, Recall:%.3f" % (precision, recall))