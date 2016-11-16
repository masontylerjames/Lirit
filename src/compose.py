import numpy as np


def outputToState(state, statematrix):
    shape = state.shape
    sample = np.random.random(shape)
    sm = sample < state
    return sm * 1
