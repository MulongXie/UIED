import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

from Config import Config
cfg = Config()


class ResClassifier():
    def __init__(self):
        self.data = None
        self.model = None

        self.image_shape = cfg.image_shape
        self.class_number = cfg.class_number
        self.class_map = cfg.class_map
        self.MODEL_PATH = cfg.MODEL_PATH

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
        self.model.save(self.MODEL_PATH)
        print("Trained model is saved to", self.MODEL_PATH)

    def load(self):
        self.model = load_model(self.MODEL_PATH)
        print('Model Loaded From', self.MODEL_PATH)

    def preprocess_img(self, image):
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255).astype('float32')
        x = np.array([x])
        return x

    def predict(self, imgs, load=False, show=False):
        """
        :type img_path: list of img path
        """
        class_names = []
        if load:
            self.load()
        if self.model is None:
            print("*** No model loaded ***")
            return
        for img in imgs:
            X = self.preprocess_img(img)
            Y = self.class_map[np.argmax(self.model.predict(X))]
            class_names.append(Y)
            if show:
                print(Y)
                cv2.imshow('element', img)
                cv2.waitKey()
        return class_names

    def evaluate(self, data, load=True):
        if load:
            self.load()
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
