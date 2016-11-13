from miditransform import midiToStateMatrix, shape
from model import model, n_steps
from os import listdir
from os.path import isfile, join, abspath


def generateXY(statematrix, offset):
    '''
    INPUT: statematrix
    OUTPUT: list of statematrix slices, list of statematrix slices
    '''
    X, Y = [], []
    i = 0
    single = [[0 for k in range(shape[1])] for j in range(shape[0])]
    for i in range(len(statematrix) / offset):
        Xi = (offset * i, n_steps + offset * i)
        Yi = (offset * (i + 1), offset * (i + 1) + n_steps)
        X_slice = statematrix[Xi[0]:Xi[1]]
        Y_slice = statematrix[Yi[0]:Yi[1]]
        if len(X_slice) < n_steps:
            X_slice += [single for n in range(n_steps - len(X_slice))]
        if len(Y_slice) < n_steps:
            Y_slice += [single for n in range(n_steps - len(Y_slice))]
        X.append(X_slice)
        Y.append(Y_slice)
    return X, Y


def getfiles(directory):
    '''
    INPUT: a directory name or list of directory names
    OUTPUT: a list of filenames
    '''
    if isinstance(directory, list):
        files = [filesfromsingledirectory(item) for item in directory]
        files = [entry for item in files for entry in item]
    else:
        files = filesfromsingledirectory(directory)
    return files


def filesfromsingledirectory(directory):
    return [abspath(directory) + '/' + f for f in listdir(directory) if isfile(join(directory, f))]
