from Lirit import Lirit
from src.miditransform import midiToStateMatrix
from src.fit import generateXY
import numpy as np


if __name__ == '__main__':
    lirit = Lirit()
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    X, Y = generateXY(sm, lirit.n_steps, lirit.offset)
    X_f = lirit._reshapeInput(np.asarray(X[:1]))
    Y_f = lirit._reshapeInput(np.asarray(Y[:1]))
    while True:
        lirit.fit(X_f, Y_f, nb_epoch=1)
