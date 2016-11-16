import numpy as np


def outputToState(output, statematrix):
    '''
    take in the output of the neural network and transform it to a suitable
    statematrix state
    '''
    # modify the probabilites of the output based on previous behavior
    conservatism = _calcConservatism(statematrix)
    state = output**conservatism

    state = _sampleByProba(state)
    return state


def generateSeed(input_shape):
    seed = np.random.random(input_shape)
    seed[:, :, 0] = (seed[:, :, 0] > .98) * 1
    seed[:, :, 1] = seed[:, :, 1] * seed[:, :, 0]
    seed = (seed > 0.5) * 1
    seed = seed[np.newaxis]
    return seed


def _sampleByProba(state):
    '''
    return a matrix of binary values based on a matrix of probabilites
    P(note) and P(actuate|note)
    '''
    state[0, :, 0] = np.random.random(len(state)) < state[0, :, 0]
    state[0, :, 1] = np.random.random(len(state)) < state[0, :, 1]
    state[0, :, 1] = state[0, :, 1] * state[0, :, 0]
    return state * 1


def _calcConservatism(statematrix):
    '''
    returns a float based on previous states that modifies note probabilities.
    returned values on (0,1) increase the probability of notes, while returned
    values on (1,inf) decrease the probability
    '''
    notes = statematrix.sum(axis=1)[:, 0]
    mean = np.mean(notes[-64:])
    return np.exp((mean - 2) / 10)
