from Lirit import Lirit
from src.miditransform import midiToStateMatrix
import numpy as np

if __name__ == '__main__':
    lirit = Lirit()
    lirit.fitmidis(
        'data/train/mozart/mz_311_1_format0.mid', nb_epoch=50)
    l = 5497 / 2
    sm = midiToStateMatrix('data/train/mozart/mz_311_1_format0.mid')
    seed = np.array(sm)[:lirit.n_steps][np.newaxis]
    lirit.save('autotestmodel.h5')
    lirit.compose(l, filename='autotest', seed=seed, verbose=True)
