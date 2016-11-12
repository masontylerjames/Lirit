from miditransform import midiToStateMatrix
from model import model, n_steps
import numpy as np

# this all needs to be rewritten

'''
To train a model

select a group of midi files to use to train

for each midi file generate a state matrix

for each state matrix make a list of inputs and a list of their corresponding outputs

stitch these lists together and that's your training corpus
'''


def getfilesfromdir(dir):
    '''
    INPUT: a directory name or list of directory names
    OUTPUT: a list of filenames
    '''

    files = None
    return files


def statematrixtoarray(statematrix):
    arr = [[[entry for note in state for entry in note]
            for state in statematrix]]
    arr = np.array(arr)
    return arr


def train(midifile):
    statematrix = midiToStateMatrix(midifile)
    return _train(statematrix)


def _train(statematrix):
    neuralnet = model()
    train = statematrixtoarray(statematrix[0: n_steps])
    test = statematrixtoarray(statematrix[1: n_steps + 1])
    neuralnet.fit(train, test)
    return neuralnet

if __name__ == '__main__':
    neuralnet = train('../data/train/mozart/mz_311_1_format0.mid')
