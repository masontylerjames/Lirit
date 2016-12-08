from keras.models import load_model
from os.path import abspath
from compose import outputToState, generateSeed
from features import addfeatures
from fit import getfiles, generateXY, fitGenerator
from miditransform import noteStateMatrixToMidi, midiToStateMatrix, shape
from model import model
import numpy as np


class Lirit(object):

    def __init__(self, n_steps=128):
        self.n_steps = n_steps
        self.model = model(self.n_steps)
        self.input_shape = (n_steps,) + shape

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def fitmidis(self, filenames, batch_size=32, **kwargs):
        X, Y = None, None
        if not isinstance(filenames, list):
            filenames = [filenames]

        generator = fitGenerator(
            filenames, self.n_steps, batch_size=batch_size)
        self.model.fit_generator(generator, **kwargs)

    def fitcollection(self, dirs, **kwargs):
        files = getfiles(dirs)
        self.fitmidis(files, **kwargs)

    def compose(self, length, filename='example', seed=None, verbose=False):
        '''
        INPUT: int

        length: length of the resulting music piece in number of 16th notes
        filename: the name of the file where the result is saved
        seed: a single input entry for the neural network to start with. If
        None a seed is randomly generated
        '''
        statematrix = None
        sm_offset = self.n_steps
        if seed is None:
            sm_offset = self.n_steps
            seed = generateSeed(self.input_shape)

        conservatism = 1.
        for state in seed:
            N_notes = state[:, 0].sum()
            conservatism = calcConservatism(N_notes, conservatism)

        inputs = addfeatures(seed)
        probas = self.model.predict(
            inputs[np.newaxis])  # generate probabilites
        # turn probas into predictions
        predict = outputToState(probas, conservatism=conservatism)
        # append predictions to statematrix
        statematrix = np.append(seed, predict, axis=0)

        while len(statematrix) < length + sm_offset:
            N_notes = predict[:, 0].sum()
            conservatism = calcConservatism(N_notes, conservatism)

            if verbose:
                print "Created {} of {} steps".format(len(statematrix) - self.n_steps, length - self.n_steps + sm_offset)
            inputs = addfeatures(
                statematrix[-self.n_steps:])[np.newaxis]
            # generate probabilites
            probas = self.model.predict(inputs)
            # turn probas into predictions
            predict = outputToState(probas, conservatism)
            # append predictions to statematrix
            statematrix = np.append(statematrix, predict, axis=0)

        noteStateMatrixToMidi(statematrix[sm_offset:], filename)

    def save(self, filename):
        self.model.save(abspath(filename))

    def load(self, filename):
        self.model = load_model(filename)


def calcConservatism(n, conservatism):
    if n < 2:
        conservatism -= .02
    conservatism += (1. - conservatism) * .3
    return conservatism
