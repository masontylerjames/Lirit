from miditransform import midiToStateMatrix
from model import model, n_steps
from os import listdir
from os.path import isfile, join, abspath

offset = n_steps / 2
'''
To train a model

select a group of midi files to use to train

for each midi file generate a state matrix

for each state matrix make a list of inputs and a list of their corresponding outputs

stitch these lists together and that's your training corpus
'''


def singletrain(filename, neuralnet):
    '''
    INPUT: string, keras model
    '''
    statematrix = midiToStateMatrix(filename)
    X, Y = generateXY(statematrix)
    neuralnet.fit(X, Y)
    pass


def generateXY(statematrix):
    '''
    INPUT: statematrix
    OUTPUT: list of statematrix slices, list of statematrix slices
    '''
    X, Y = [], []
    i = 0
    while True:
        try:
            Yi = (offset + n_steps * i, offset + n_steps * (i + 1))
            Xi = (n_steps * i, n_steps * (i + 1))
            Y.append(statematrix[Yi[0]:Yi[1]])
            X.append(statematrix[Xi[0]:Xi[1]])
        except IndexError:
            break
        i += 1
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
    return [abspath(directory) + f for f in listdir(directory) if isfile(join(directory, f))]

if __name__ == '__main__':
    neuralnet = model()
    singletrain(
        '../data/train/mozart/mz_311_1_format0.mid', neuralnet)
