from src.miditransform import lowerBound
import numpy as np

n_features = 173


def addfeatures(statematrix):
    return np.array([manufacturestate(state) for state in statematrix])


def manufacturestate(state):
    return [manufacturefeatures(note, i, state) for i, note in enumerate(state)]


def manufacturefeatures(note, i, state):
    pitch_class = [0] * 12
    pitch = lowerBound + i
    pitch_class[pitch % 12] = 1
    pitches = [0] * 12
    for i, n in enumerate(state[:, 0]):
        if n == 1:
            pitches[(i + lowerBound) % 12] += 1
    key_scores = keyscores(state, np.array(pitches))
    vicinity = get_vicinity(i, state)
    return note.tolist() + [pitch] + pitch_class + pitches + key_scores + vicinity


def keyscores(state, pitches):
    major = np.array([1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1])
    minor = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1])
    har_minor = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1])
    mel_minor = np.array([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1])
    major_keys = [pitches.dot(
        np.append(major[i:], major[:i])) for i in range(12)]
    minor_keys = [pitches.dot(
        np.append(minor[i:], minor[:i])) for i in range(12)]
    har_minor_keys = [pitches.dot(
        np.append(har_minor[i:], har_minor[:i])) for i in range(12)]
    mel_minor_keys = [pitches.dot(
        np.append(mel_minor[i:], mel_minor[:i])) for i in range(12)]
    return major_keys + minor_keys + har_minor_keys + mel_minor_keys


def get_vicinity(i, state):
    vicinity = []
    for j in range(-24, 25):
        try:
            vicinity += state[i + j, :].tolist()
        except:
            vicinity += [0, 0]
    return vicinity
