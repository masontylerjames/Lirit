import numpy as np


def outputToState(output):
    '''
    take in the output of the neural network and transform it to a suitable
    statematrix state
    '''
    # modify the probabilites of the output based on previous behavior
    conservatism = _calcConservatism()
    state = output**conservatism

    state = _sampleByProba(state)
    return state


def generateSeed():
    pass


def _sampleByProba(state):
    '''
    return a matrix of binary values based on a matrix of probabilites
    P(note) and P(actuate|note)
    '''
    state[:, 0] = np.random.random(len(state)) < state[:, 0]
    state[:, 1] = np.random.rnaomd(len(state)) < state[:, 1]
    state[:, 1] = state[:, 1] * state[:0]
    return state * 1


def _calcConservatism():
    '''
    returns a float based on previous states that modifies note probabilities.
    returned values on (0,1) increase the probability of notes, while returned
    values on (1,inf) decrease the probability
    '''
    return 1
