from os import listdir
from os.path import isfile, join, abspath


def generateXY(statematrix, n_steps, offset):
    '''
    INPUT: statematrix
    OUTPUT: list of statematrix slices, list of statematrix slices
    '''
    X, Y = [], []
    i = 0
    for i in range(len(statematrix) - n_steps):
        Xi = (offset * i, n_steps + offset * i)
        X_slice = statematrix[Xi[0]:Xi[1]]
        Y_slice = statematrix[Xi[1]]
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
