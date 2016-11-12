from miditransform import midiToStateMatrix
from model import model, n_steps
import numpy as np
from os import listdir
from os.path import isfile, join, abspath
# this all needs to be rewritten

'''
To train a model

select a group of midi files to use to train

for each midi file generate a state matrix

for each state matrix make a list of inputs and a list of their corresponding outputs

stitch these lists together and that's your training corpus
'''


def masstrain():
    pass


def getfilesfromdir(directory):
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
    pass
