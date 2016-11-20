from features import noteStateMatrixToInputForm, features_shape
from os import listdir
from os.path import isfile, join, abspath
import numpy as np


def generateInputsAndTargets(statematrix, n_steps):
    '''
    INPUT: numpy array, int

    statematrix is an (n, 87, 2) array that contains all the pitch
    on/off and actuation data from a midi file
    n_steps is an int that describes the number of time steps put into an input
    slice
    '''
    Y = []
    steps = range(len(statematrix) - n_steps)
    Y = [statematrix[n_steps + i] for i in steps]
    inputs = generateInputs(statematrix, n_steps)
    return inputs, np.asarray(Y)


def generateInputs(statematrix, n_steps):
    X, beat = [], []
    steps = range(len(statematrix) - n_steps)
    for t in range(len(statematrix)):
        mods = [(t // i) % 2 for i in [1, 2, 4, 8]]
        beat.append([2 * x - 1 for x in mods])
    X = [statematrix[i:n_steps + i] for i in steps]
    beats = [beat[i:n_steps + i] for i in steps]
    inputs = [np.asarray(X), np.asarray(beats)]
    return inputs


def generateXY(statematrix, n_steps, offset, features=False):
    '''
    INPUT: statematrix
    OUTPUT: list of statematrix slices, list of statematrix slices
    '''
    if offset == 0:
        offset = 1
    X, Y = [], []
    feature_matrix = np.asarray(
        noteStateMatrixToInputForm(statematrix)) if features else statematrix
    steps = range(len(statematrix) - n_steps)
    X = np.asarray(
        [feature_matrix[offset * i:n_steps + offset * i] for i in steps])
    Y = np.asarray([statematrix[n_steps + offset * i] for i in steps])
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
