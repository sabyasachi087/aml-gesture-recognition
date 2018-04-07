import pandas as pd
import numpy as np
from DataType import DataType

columns = ["cord_x", "cord_y", "cord_z", "tilt_x", 'tilt_y', 'tilt_z', 'thumb', 'index', 'middle', 'ring', 'little']
SPLIT_MARKER = '---------END--------'


def readGesture(gesture, type=DataType.test):
    if type == DataType.test:
        return pd.read_csv('test/' + str(gesture) + '.txt', names=columns)
    elif type == DataType.train:
        return pd.read_csv('data/' + str(gesture) + '.txt', names=columns)
    else:
        raise Exception("Invalid type ")


def splitData(data):
    training_list = [];train_data = pd.DataFrame(columns=columns, dtype=np.float64)
    for row in range(len(data)):
        row_data = data.iloc[row].copy()
        if row_data.cord_x == SPLIT_MARKER:
            train_data.dropna(axis=0, inplace=True)
            training_list.append(train_data)
            train_data = pd.DataFrame(columns=columns, dtype=np.float64)
        else:
            train_data.loc[len(train_data)] = row_data
    return training_list


def getAllDataFileName(type=DataType.train):
    from os import listdir
    from os.path import isfile, join
    if type == DataType.test:
        DATA_PATH = 'test/'
    else:
        DATA_PATH = 'data/'
    onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    names = []
    for file in onlyfiles:
        names.append(file.split(".")[0])
    return names


def getData(gesture='wrong', type=DataType.train):
    raw_gesture_data = readGesture(gesture, type=DataType.train)
    return splitData(raw_gesture_data)


def getTrainData():
    train_file = getAllDataFileName(DataType.train)
    X_train = [];y_train = []
    for gesture in train_file:
        for data in getData(gesture):
            X_train.append(data)
            y_train.append(gesture)
    return X_train, y_train


def getTestData():
    test_file = getAllDataFileName(DataType.test)
    X_test = [];y_test = []
    for gesture in test_file:
        for data in getData(gesture):
            X_test.append(data)
            y_test.append(gesture)
    return X_test, y_test

