from keras.models import Model
from keras.layers import Flatten, Input, merge, RepeatVector, Reshape, TimeDistributed
from keras.layers import Bidirectional, Dense, Lambda, GRU
import keras.backend as K
import numpy as np
from Lirit.miditransform import midiToStateMatrix
from Lirit.fit import generateXY

base_shape = (88, 2)
base_latent_dim = 64
'''
TO DO:
add back end of an autoencoder for the statematrix series
    not using a VAE because I don't need to generate the series
    also why I'm not making a discriminator for the AE output

Possibles:
send series latent vector to discriminator instead of series
use convolutions

Wacky Idea:
Turn algorithm into an action generator that is then used to
'''


class Lirit(object):

    def __init__(self):
        pass

    def ae_test(self, n, m, **kwargs):
        autoencoder = self.ae(64, (88, 2), n, m)
        f = "data/train/mozart/mz_311_1_format0.mid"
        sm = midiToStateMatrix(f)
        X, Y = generateXY(sm, 64)

        autoencoder.compile(loss='binary_crossentropy',
                            optimizer='rmsprop')
        autoencoder.fit(X, X, **kwargs)
        return X, autoencoder.predict(X)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=z_mean._keras_shape, mean=0., std=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def net_trainable(net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def train_vae_gan(self, n_steps, input_shape, latent_dim_state, latent_dim_series):
        ae_model = self.ae_front_half(
            n_steps, input_shape, latent_dim_state, latent_dim_series)
        vae = self.vae_front_half(input_shape, latent_dim_state)
        gen = self.generator(n_steps, input_shape, latent_dim_state)
        dis = self.discriminator(n_steps, input_shape)

        series = Input(shape=(n_steps,) + input_shape)
        statematrix = Input(shape=input_shape)

        y = vae(statematrix)
        mean = y[1]
        log_var = y[2]

        z = ae_model(series)
        gen_out = gen([y[0], z])
        dis_class = dis([series, gen_out[0]])

        full_model = Model(input=[series, statematrix], output=[
                           mean, log_var, gen_out[0], gen_out[1], dis_class])
        '''
        note: a model needs to be recompiled for changes in trainability to mean anything

        begin loop:
            generate examples from generator using a random latent vector and a random selected input clip
            make discriminator trainable
            compile discriminator
            train only the discriminator on equal parts generated data and actual data
            freeze the discriminators ability to train
            compile full model
            train the full model on actual data
        '''
        return full_model

    def generator(self, n_steps, input_shape, latent_dim_state, latent_dim_series):
        '''
        Returns a model that takes in a latent vector and a series of state matrices and returns the next statematrix in the series
        '''
        latent_state_vector = Input(latent_dim_state)
        latent_series_vector = Input(latent_dim_series)

        gen = self.generator_core(n_steps, input_shape,
                                  latent_dim_state, latent_dim_series)
        ae = self.ae_front_half(
            n_steps, input_shape, latent_dim_series)

        x = ae(latent_series_vector)
        y = gen([latent_state_vector, x])
        return Model(input=[latent_state_vector, latent_series_vector], output=y)

    def generator_core(n_steps, input_shape, latent_dim_state, latent_dim_series):
        '''
        returns a generative model from latent vector and series input to output
        '''
        latent_state_vector = Input(shape=(latent_dim_state,))
        s_layer = Input(shape=(latent_dim_series,))
        merge_layer = merge(
            [latent_state_vector, s_layer], mode='concat')
        layer = Dense(np.prod(input_shape),
                      activation='sigmoid')(merge_layer)
        layer = Reshape(input_shape)(layer)
        return Model(input=[latent_state_vector, s_layer], output=[layer, s_layer])

    def vae_front_half(self, input_shape, latent_dim_state):
        '''
        returns the portion of a VAE up to the latent vector as a model
        '''
        input_layer = Input(shape=input_shape)
        layer = Flatten()(input_layer)
        mean = Dense(latent_dim_state)(layer)
        log_var = Dense(latent_dim_state)(layer)
        latent_vector = Lambda(self.sampling, output_shape=(
            latent_dim_state,))([mean, log_var])
        return Model(input=[input_layer], output=[latent_vector, mean, log_var])

    def ae(self, n_steps, input_shape, latent_dim_state, latent_dim_series):
        '''
        returns an autoencoder training model for
        '''
        series = Input(shape=(n_steps,) + input_shape)
        ae_fh = self.ae_front_half(n_steps, input_shape,
                                   latent_dim_state, latent_dim_series)
        ae_bh = self.ae_back_half(n_steps, input_shape,
                                  latent_dim_state, latent_dim_series)
        y = ae_fh(series)
        y = ae_bh(y)
        return Model(input=series, output=y)

    def ae_front_half(n_steps, input_shape, latent_dim_state, latent_dim_series):
        '''
        Returns a model that turns an input series of statematrices into a latent vector
        '''
        series = Input(shape=(n_steps,) + input_shape)
        layer = TimeDistributed(Flatten())(series)
        layer = TimeDistributed(Dense(latent_dim_state))(layer)
        layer = GRU(latent_dim_series)(layer)
        return Model(input=[series], output=[layer])

    def ae_back_half(n_steps, input_shape, latent_dim_state, latent_dim_series):
        '''
        Returns a model that turns a latent vector into the corresponding series of statematrices
        '''
        latent_series_vector = Input(shape=(latent_dim_series,))
        layer = RepeatVector(n_steps)(latent_series_vector)
        layer = Bidirectional(
            GRU(latent_dim_state * 4, return_sequences=True), merge_mode='ave')(layer)
        layer = Bidirectional(
            GRU(latent_dim_state, return_sequences=True), merge_mode='ave')(layer)
        layer = TimeDistributed(Dense(np.prod(input_shape)))(layer)
        layer = TimeDistributed(Reshape(input_shape))(layer)
        return Model(input=[latent_series_vector], output=[layer])

    def discriminator(n_steps, input_shape):
        '''
        returns a model that takes in a series and a
        '''
        series = Input(shape=(n_steps,) + input_shape)
        generated = Input(shape=input_shape)
        merge_layer = merge([series, generated],
                            mode='concat', concat_axis=1)
        layer = TimeDistributed(Flatten())(merge_layer)
        layer = GRU(32)(layer)
        layer = Dense(1)(layer)
        return Model(input=[series, generated], output=[layer])
