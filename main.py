from Lirit import Lirit
from src.miditransform import midiToStateMatrix
import numpy as np

if __name__ == '__main__':
    lirit = Lirit()
    lirit.load('data/models/newtest.h5')
    l = 5497 / 2
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    seed = np.array(sm)[:256][np.newaxis]
    lirit.compose(l, filename='test', seed=seed, verbose=True)
