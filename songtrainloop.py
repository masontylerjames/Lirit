from Lirit import Lirit


if __name__ == '__main__':
    lirit = Lirit()
    while True:
        lirit.fitmidis(
            'data/train/mozart/mz_311_1_format0.mid', nb_epoch=1)
