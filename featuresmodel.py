from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from src.miditransform import midiToStateMatrix, lowerBound, upperBound
from src.fit import *
import numpy as np
import tensorflow as tf


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


def model():
    n_steps = 128
    n = 173
    dropout = 0.5
    input_layer = Input(shape=(n_steps, 87, n))
    permute_1 = Permute((2, 1, 3))(input_layer)
    with tf.device('/gpu:0'):
        per_pitch_lstm_1 = TimeDistributed(
            LSTM(256, return_sequences=True), name='pitch_lstms_1')(permute_1)
        dropout_1 = Dropout(dropout)(per_pitch_lstm_1)
    with tf.device('/gpu:1'):
        per_pitch_lstm_2 = TimeDistributed(
            LSTM(256), name='pitch_lstms_2')(dropout_1)
        dropout_2 = Dropout(dropout)(per_pitch_lstm_2)
    with tf.device('/gpu:2'):
        pitch_lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(
            dropout_2)
        dropout_3 = Dropout(dropout)(pitch_lstm_1)
    with tf.device('/gpu:3'):
        pitch_lstm_2 = Bidirectional(
            LSTM(64, return_sequences=True))(dropout_3)
        dropout_4 = Dropout(dropout)(pitch_lstm_2)
    with tf.device('/gpu:4'):
        dense_1 = TimeDistributed(Dense(32))(dropout_4)
        dense_2 = TimeDistributed(Dense(2))(dense_1)
        out = Activation('sigmoid')(dense_2)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def run(**kwargs):
    X, Y = None, None

    filepath = "data/models/features2/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    dirs = ['data/train/mozart', 'data/train/bach',
            'data/train/beethoven', 'data/train/chopin', 'data/train/tchaikovsky']
    files = getfiles(dirs)
    for f in files:
        print 'loading and engineering {}'.format(f)
        sm = midiToStateMatrix(f)
        if sm is not None:
            smf = addfeatures(sm)
            _, Y_f = generateInputsAndTargets(sm, 128)
            X_f, _ = generateInputs(smf, 128)
            if X is None or Y is None:
                X, Y = X_f, Y_f
            else:
                try:
                    X = np.append(X, X_f, axis=0)
                    Y = np.append(Y, Y_f, axis=0)
                except MemoryError:
                    print 'MemoryError, bailing on further pieces'
                    break
    print 'Building model'
    nn = model()
    nn.fit(X, Y, validation_split=0.2,
           callbacks=[checkpoint], **kwargs)
    return nn

if __name__ == '__main__':
    pass
