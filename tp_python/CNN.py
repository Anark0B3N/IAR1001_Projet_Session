from multiprocessing.spawn import prepare

import numpy
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

from ImageDetails import IMG_DETAILS

training_data = []
training_data_label = []
imgTable = []

def load_imgs():
    listOfImages = []


    myimg = cv2.imread("digits.png", -1)
    imageFaitParTom = cv2.imread("test-nb-2.png", -1)
    imgray = cv2.cvtColor(imageFaitParTom, cv2.COLOR_BGR2GRAY)

    imgTest = np.array(imgray).flatten()
    cells = [np.hsplit(row, 100) for row in np.vsplit(myimg, 50)]

    # Make it into a Numpy array: its size will be (50,100,20,20)
    global imgTable
    imgTable = np.array(cells)



def create_training_data():

    global imgTable
    global training_data
    global training_data_label

    label_num = -1
    for index, img_row in enumerate(imgTable):  # do dogs and cats

        if index % 5 == 0:
            # set label
            label_num += 1

        for img in img_row:
            img_resized = cv2.resize(img, (IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))
            img_resized = img_resized / 255
            img_resized = img_resized.reshape(IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE)
            training_data.append(img_resized)
            training_data_label.append(label_num)


    # training_data shape is (5000 images, 50 pixels, 50 pixels
    # training_data_label shape is (5000 label, 10 float representing the number) -> ex: 5 = [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

    # each number have 500 images for a total of 5000


    X = np.array(training_data).reshape(5000, IMG_DETAILS.IMG_SIZE*IMG_DETAILS.IMG_SIZE)
    # training_data_label = np.array(training_data_label).reshape(-1, 10)

    train(X, numpy.array(training_data_label))


def train(X, label):
    (X1_train, y_train), (X1_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = numpy.empty((60000, IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))
    X_test = numpy.empty((60000, IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))

    it_image = 0
    for image in X1_train:
        X_train[it_image] = cv2.resize(image, (IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))
        it_image += 1

    it_image = 0
    for image in X1_test:
        X_test[it_image] = cv2.resize(image, (IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))
        it_image += 1


    X_train = X_train / 255
    X_test = X_test / 255

    X_train_flattened = X_train.reshape(len(X_train), IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE)
    np.append(X_train_flattened, X)
    np.append(y_train, label)
    X_test_flattened = X_test.reshape(len(X_test), IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE)

    it = 0
    for img in X_train_flattened:
        imgCopy = img
        it2 = 0
        for pixel in imgCopy:
            if pixel > 0.2:
                imgCopy[it2] = 1
            else:
                imgCopy[it2] = 0
            it2 += 1

        X_train_flattened[it] = imgCopy



    it = 0
    for img in X_test_flattened:
        imgCopy = img
        it2 = 0
        for pixel in imgCopy:
            if pixel > 0.2:
                imgCopy[it2] = 1
            else:
                imgCopy[it2] = 0
            it2 += 1

        X_test_flattened[it] = imgCopy


    model = Sequential([
        Dense(100, input_shape=(IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE,), activation='relu'),
        Dense(2500, activation='relu'),
        Dense(2000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(500, activation='relu'),
        Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_flattened, y_train, epochs=7)


    # model = Sequential()
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    #
    # model.add(Dense(64))
    #
    # model.add(Dense(10))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # model.fit(X, label, batch_size=32, epochs=3, validation_split=0.01)
    #
    model.save('cnn_image_digit_model.model')


def test_model():
    global training_data

    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # X_train = X_train / 255
    # X_test = X_test / 255

    # X_train_flattened = X_train.reshape(len(X_train), IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE)
    # X_test_flattened = X_test.reshape(len(X_test), IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE)

    myModel = tf.keras.models.load_model('cnn_image_digit_model.model')

    appImg = cv2.imread("handwritten_input.png", cv2.COLOR_BGR2GRAY)
    # imgray = cv2.cvtColor(appImg, cv2.COLOR_BGR2GRAY)
    testImg = cv2.resize(appImg, (IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))

    cv2.imshow('sds', testImg)
    cv2.waitKey(0)
    # cv2.imshow('sds', X_test[0])
    # cv2.waitKey(0)

    # imgTestSave = X_test[0]
    # x = 0
    # for row in imgTestSave:
    #     y = 0
    #     for pixel in row:
    #         if pixel > (0.2 * 255):
    #             imgTestSave[x][y] = 255
    #         else:
    #             imgTestSave[x][y] = 0
    #         y += 1
    #     x += 1


    # cv2.imwrite("testmnist.png", imgTestSave)

    testImg = testImg.reshape(IMG_DETAILS.IMG_SIZE * IMG_DETAILS.IMG_SIZE)
    testImg = testImg / 255

    testArray = numpy.array([testImg, testImg])

    # global imgTable
    # global training_data
    # global training_data_label
    #
    # load_imgs()
    #
    # label_num = -1
    # for index, img_row in enumerate(imgTable):  # do dogs and cats
    #
    #     if index % 5 == 0:
            # set label
            # label_num += 1
        #
        # for img in img_row:
        #     img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #     img_resized = img_resized / 255
        #     img_resized = img_resized.reshape(IMG_SIZE * IMG_SIZE)
        #     training_data.append(img_resized)
        #     training_data_label.append(label_num)

    # training_data shape is (5000 images, 50 pixels, 50 pixels
    # training_data_label shape is (5000 label, 10 float representing the number) -> ex: 5 = [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

    # each number have 500 images for a total of 5000

    # X = np.array(training_data).reshape(5000, IMG_SIZE * IMG_SIZE)
    # training_data_label = np.array(training_data_label).reshape(-1, 10)

    # labels = numpy.array(training_data_label)

    y_predicted = myModel.predict(testArray)

    # it = 0
    # for item in y_predicted:
    #     print("should be :")
    #     print(labels[it])
    #     print(item)
    #     print(np.argmax(item))
    #     it += 1

    print(y_predicted[0])
    print(np.argmax(y_predicted[0]))
