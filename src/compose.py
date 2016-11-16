import numpy as np


def outputToState(output):
    state = output
    state[:, 0] = np.random.random(len(state)) < state[:, 0]
    state[:, 1] = np.random.rnaomd(len(state)) < state[:, 1]
    state[:, 1] = state[:, 1] * state[:0]
    return state * 1
