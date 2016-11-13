from src.compose import compose
from src.train_model import multitrain
from src.model import model


class Lirit(object):

    def __init__(self):
        self.model = model()

    def compose(self, length, filename='example', seed=None):
        compose(self.model, length, filename=filename, seed=seed)

if __name__ == '__main__':
    neuralnet = model()
    multitrain('data/train/mozart/', neuralnet)
    compose(neuralnet, 0)
