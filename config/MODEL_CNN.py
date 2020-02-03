from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np

from config.CONFIG import Config
cfg = Config()


class CNN:

    def __init__(self):
        self.data = None
        self.model = None

        self.image_shape = cfg.image_shape
        self.class_number = cfg.class_number
        self.class_map = cfg.class_map
        self.MODEL_PATH = cfg.MODEL_PATH

    def network(self):
        # block 1
        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self.image_shape, padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        # block 2
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        # block 3
        self.model.add(Dense(self.class_number, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model.fit(self.data.X_train, self.data.Y_train, batch_size=64, epochs=20, verbose=1, validation_data=(self.data.X_test, self.data.Y_test))

    def train(self, data):
        self.data = data
        self.model = Sequential()
        self.network()
        self.model.save(self.MODEL_PATH)
        print("Trained model is saved to", self.MODEL_PATH)

    def evaluate(self, data, load=True):
        # calculate TP, FN, FP, TN
        def calculate_n_p(matrix):
            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(len(matrix)):
                TP += matrix[i][i] / np.sum(matrix[i])
                FN += (np.sum(matrix[:, i]) - matrix[i][i]) / np.sum(matrix[:, i])
                FP += (np.sum(matrix[i]) - matrix[i][i]) / np.sum(matrix[i])
                TN += (np.trace(matrix) - matrix[i][i]) / np.trace(matrix)
            TP = TP / len(matrix)
            FN = FN / len(matrix)
            FP = FP / len(matrix)
            TN = TN / len(matrix)
            return TP, FN, FP, TN

        if load:
            self.load()
        X_test = data.X_test
        Y_test = [np.argmax(y) for y in data.Y_test]
        Y_pre = []
        for X in X_test:
            X = np.array([X])
            Y_pre.append(np.argmax(self.model.predict(X)))

        matrix = confusion_matrix(Y_test, Y_pre)
        TP, FN, FP, TN = calculate_n_p(matrix)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        balanced_accuracy = TP
        print(matrix)
        print('\nTP:%.3f \t FN:%.3f \nFP:%.3f \t TN:%.3f\n' % (TP, FN, FP, TN))
        print('recall:%.3f \t precision:%.3f \t accuracy:%.3f \t balanced accuracy:%.3f' % (recall, precision, accuracy, balanced_accuracy))

    def predict(self, imgs, load=False, show=False):
        """
        :type imgs: list of imgs
        """
        prediction = []
        if load:
            self.load()
        for img in imgs:
            X = cv2.resize(img, self.image_shape[:2])
            X = np.array([X])  # from (64, 64, 3) to (1, 64, 64, 3)
            Y = self.class_map[np.argmax(self.model.predict(X))]
            prediction.append(Y)
            if show:
                print(Y)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return prediction

    def load(self):
        self.model = load_model(self.MODEL_PATH)
        print('Model Loaded From', self.MODEL_PATH)
