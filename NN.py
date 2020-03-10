# -*- coding: utf-8 -*-
"""
Neural network replacing abaqus solver

Created on June 19 13:44:13 2019

Python 3.7. 

@author: M.Cadek
"""

import numpy as np
from tqdm import tqdm
import dataio
import time
from tflearn import DNN
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


def getTrainingData(f):
    _input_data = []
    _output_data = []
    data, keys = dataio.readFile(f)

    # keys = ['vf','nu_f','ar','E_f','E_m','nu_m', 'Stress']

    for k in keys:
        if 'Stress' in k:
            _output_data = data[k]
        else:
            _input_data.append(data[k])

    _input_data = np.array(_input_data)
    input_data = np.swapaxes(_input_data, 0, 1)
    # input_data = input_data.tolist()
    print(input_data[:12])
    print(np.shape(input_data))
    # input_data = np.ravel(input_data)
    # output_data = np.array(_output_data)

    return input_data, _output_data


def testNetworkTF(input_data, filename, model=None):
    _input_data = input_data
    _output_data = []

    print("--- loading model ---")
    # _model.load(filename)
    if model != None:
        _model = model
    else:
        _model = create_modelTF()
        #         _model.load("D:/Python3/snake_NN/snake_nn.tfl", weights_only=True)
        _model.load("/home/cada/PycharmProjects/NN/" + filename)

    n = 0
    # while n < len(_input_data):
    predictions = _model.predict(_input_data)
    # print(input_vect)
    # print(output_vect)
    _output_data.extend(predictions)

    # n += 1

    return _input_data, _output_data


def trainNetworkTF(x, y, model, filename):
    # X = np.array([i[0] for i in x]).reshape(-1, 6, 1)
    # Y = np.array([i[0] for i in y]).reshape(-1, 1)

    x = x.reshape([-1, 1, 6, 1])
    y = np.array(y)
    y = y.reshape([-1, 1])

    # print(np.shape(x[0]))
    # x = x.reshape(1,6,1)

    model.fit(x, y, n_epoch=20, shuffle=True, run_id=filename)
    print("--- saving trained model ---")
    model.save(filename)
    return model


def create_modelTF():
    network = input_data(shape=[None, 1, 6, 1], name='input')
    network = fully_connected(network, 20, activation='relu')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
    #     model = DNN(network, checkpoint_path='snake_nn.tfl', tensorboard_dir='log', max_checkpoints=1 )
    model = DNN(network, tensorboard_dir='log')
    return model


def main():
    NN_fileName = "nn_model.tfl"
    testData = np.array([0.5, 0.15000000000000002, 7.3003, 28000.0, 10000, 0.4])
    testData = testData.reshape(-1, 1, 6, 1)

    train = 0
    #
    if train:
        inputData, outputData = getTrainingData('FE_META.out')
        modelTF = create_modelTF()
        trained_model = trainNetworkTF(inputData, outputData, modelTF, NN_fileName)
    else:
    # time.sleep(5)
        inputTestData, outputTestData = testNetworkTF(testData, NN_fileName)

        print(inputTestData, outputTestData)
        print(outputTestData[0])
        print(1-abs(outputTestData[0]/1053))
    # print("max score: ", max(scoreTest))
    # print(len(inputTestData))
    # print(len(outputTestData))


if __name__ == "__main__":
    main()
