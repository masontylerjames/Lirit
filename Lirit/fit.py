from os import listdir
from os.path import isfile, join, abspath
import numpy as np


def generateXY(statematrix, n_steps,):
    '''
    INPUT: statematrix
    OUTPUT: list of statematrix slices, list of statematrix slices
    '''
    X, Y = [], []
    steps = range(len(statematrix) - n_steps)
    X = np.asarray([statematrix[i:n_steps + i] for i in steps])
    Y = np.asarray([statematrix[n_steps + i] for i in steps])
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
