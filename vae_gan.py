from keras.layers import Input
from keras.models import Model
from vae_gan_models import vae_front_half, generator, discriminator, input_shape

'''
Possibles:
send series latent vector to discriminator instead of series
use convolutions

Wacky Idea:
Turn algorithm into an action generator that is then used in a QNet
'''


class Lirit(object):
    '''
    docstring
    '''

    def __init__(self, batch_size, n_steps, latent_dim_state, latent_dim_series, std=1.0):
        '''
        docstring
        '''
        self.batch_size = batch_size  # batch_size needs to be set for VAE
        self.n_steps = n_steps  # number of steps in the series input
        # number of latent dimensions to use for an individual state
        self.latent_dim_state = latent_dim_state
        # numer of latent dimensions to use for a series
        self.latent_dim_series = latent_dim_series

        self.std = std  # target standard deviation for VAE

        self.state = Input(batch_shape=(
            self.batch_size,) + self.input_shape, name='sm_input')
        self.series = Input(batch_shape=(self.batch_size, self.n_steps) +
                            input_shape, name='series_input')
        self.latent_vector = Input(batch_shape=(
            self.batch_size, self.latent_dim_state), name='latent_vector')

        self.encoder = vae_front_half(batch_size, latent_dim_state)
        self.generator = generator(
            batch_size, n_steps, latent_dim_state, latent_dim_series)
        self.discriminator = discriminator(batch_size, n_steps)

    def _net_trainable(net, val):
        '''
        docstring
        '''
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def train_vae_gan(self):
        '''
        docstring
        '''

        # assemble and train discriminator with true/noise
        self._train_discriminator()

        # assemble and train VAE
        self._train_vae()

        # assemble generator and discriminator, train generator
        self._train_generator()

        pass

    def _train_discriminator(self):
        # make sure discriminator is trainable
        # generate n false examples and their labels with generator
        # pick n true examples with proper labels
        # fit discriminator for some time
        pass

    def _train_vae(self):
        # pick n true examples
        # assemble vae
        # fit for some time
        pass

    def _train_generator(self):
        # make discriminator not trainable
        # assemble gan
        # fit gan
        pass
