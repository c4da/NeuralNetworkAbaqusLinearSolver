import numpy as np


def readFile(fileName):
    data = {}
    heading = []
    f = open(fileName)
    lines = f.readlines()
    for i, line in enumerate(lines):
        lineList = line.strip().split(',')
        if i == 0:
            heading = lineList
        else:
            for j, h in enumerate(heading):
                # print(h)
                if h not in data:
                    data[h] = []
                else:
                    data[h].append(float(lineList[j]))

    for k in heading:
        mean = np.mean(data[k])
        std = np.std(data[k])
        # print(mean, std)
        data[k] = (data[k] - mean)/std

    return data, heading


def norm(x):
    mean = np.mean(data[k])
    std = np.std(data[k])
    return (x - mean) / std


if __name__ == '__main__':
    data, keys = readFile('FE_META.out')
    # print(data.keys())
    _output_data =[]
    _input_data = []
    # keys = ['vf','nu_f','ar','E_f','E_m','nu_m', 'Stress']

    for k in keys:
        if 'Stress' in k:
            _output_data = data[k]
        else:
            _input_data.append(data[k])

    _input_data = np.array(_input_data)
    _input_data = np.swapaxes(_input_data, 0, 1)
    # _input_data = np.ravel(_input_data)
    # input_data = _input_data.tolist()
    # print(np.shape(np.array(_input_data)))
    # input_data = np.reshape(_input_data, 85535*6, order='F')

    # input_data = np.array([i[1] for i in _input_data]).reshape(-1, 6, 1)
    # print(_input_data)
    # X = np.array([i[0] for i in input_data])
    X =_input_data.reshape([-1, 1, 6, 1])
    # print(input_data[:12])
    print(np.shape(X[0]))
    print(np.shape(X))
    print(len(X))
    print(X[0])
    testData = np.array([0.5, 0.15000000000000002, 7.3003, 28000.0, 10000, 0.4])
    testData = testData.reshape([-1, 6, 1])
    # print(_output_data[:12])