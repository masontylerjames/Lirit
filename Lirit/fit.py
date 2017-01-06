from os import listdir
from os.path import isfile, join, abspath
from miditransform import midiToStateMatrix
from features import addfeatures
import cPickle as pickle
import numpy as np
import os


def fitGenerator(files, n_steps, batch_size=32, verbose=True):
    '''
    Inputs: list of files to build into statematrices, number of steps to feed into the network

    Takes a list of midi files, turns them into state and feature matrices, and then pickles those matrices to be loaded later as part of the generator
    '''
    X, Y = None, None
    pickles = []
    for f in files:
        filename = 'temp/' + \
            f.split('/')[-1].split('.')[0] + '.pkl'
        d = os.path.dirname(filename)
        if verbose:
            print 'processing {}'.format(f)
        if not os.path.exists(filename):
            statematrix = midiToStateMatrix(f)
            if statematrix is not None:
                featuresmatrix = addfeatures(statematrix)
                if not os.path.exists(d):
                    os.makedirs(d)
                with open(filename, 'w') as out:
                    pickle.dump((featuresmatrix, statematrix), out)
                pickles.append(filename)

    if pickles:
        while True:
            for p in pickles:
                if verbose:
                    print "Using file {}".format(p)

                featuresmatrix, statematrix = pickle.load(p)
                for i in range(len(statematrix) - n_steps):
                    X_slice = featuresmatrix[i:n_steps + i]
                    Y_slice = statematrix[n_steps + i]
                    if X is None:
                        X, Y = X_slice, Y_slice
                    else:
                        X = np.append(X, X_slice, axis=0)
                        Y = np.append(Y, Y_slice, axis=0)
                    if len(X) == batch_size:
                        yield X, Y
                        X, Y = None, None
    else:
        raise Exception('No valid files submitted')


def generateXY(statematrix, n_steps):
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
