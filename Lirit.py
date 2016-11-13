from src.compose import compose
from src.train_model import multitrain
from src.model import model

if __name__ == '__main__':
    neuralnet = model()
    multitrain(neuralnet)
    compose(model, 0)
