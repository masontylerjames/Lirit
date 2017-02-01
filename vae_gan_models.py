from keras.layers import Input, merge, Dense, Reshape, Flatten, Lambda, TimeDistributed, GRU
from keras.models import Model
import keras.backend as K
import numpy as np

input_shape = (88, 2)


def _sampling(self, X, std):
    '''
    docstring
    '''
    z_mean, z_log_sigma = X
    epsilon = K.random_normal(
        shape=K.shape(z_mean), mean=0., std=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon


def vae_front_half(batch_size, latent_dim_state):
    '''
    returns the portion of a VAE up to the latent vector as a model
    '''
    input_layer = Input(batch_shape=(
        batch_size,) + input_shape)
    layer = Flatten()(input_layer)
    mean = Dense(latent_dim_state)(layer)
    log_var = Dense(latent_dim_state)(layer)
    latent_vector = Lambda(_sampling, output_shape=(
        latent_dim_state,))([mean, log_var])
    return Model(input=[input_layer], output=[latent_vector, mean, log_var])


def generator(batch_size, n_steps, latent_dim_state, latent_dim_series):
    '''
    Returns a model that takes in a latent vector and a series of state matrices and returns the next
    statematrix in the series
    '''
    latent_state_vector = Input(batch_shape=(
        batch_size, latent_dim_state))
    series = Input(batch_shape=(
        batch_size, n_steps) + input_shape)

    gen = _generator_core(latent_dim_state, latent_dim_series)
    ae = _ae_front_half(batch_size, n_steps,
                        latent_dim_state, latent_dim_series)

    x = ae(series)
    y = gen([latent_state_vector, x])
    return Model(input=[latent_state_vector, series], output=y)


def discriminator(batch_size, n_steps):
    '''
    returns a model that takes in a series and a generated statematrix and decides if the generated
    statematrix is real or not
    '''
    series = Input(batch_shape=(batch_size,
                                n_steps) + input_shape)
    generated = Input(batch_shape=(
        batch_size,) + input_shape)
    generated_shape = Reshape((1,) + input_shape)(generated)
    merge_layer = merge([series, generated_shape],
                        mode='concat', concat_axis=1)
    layer = TimeDistributed(Flatten())(merge_layer)
    layer = GRU(32)(layer)
    layer = Dense(1)(layer)
    return Model(input=[series, generated], output=[layer])


def _generator_core(latent_dim_state, latent_dim_series):
    '''
    returns a model that outputs the input latent series vector and the combination of the latent
    series vector and the latent state vector
    '''
    latent_state_vector = Input(shape=(latent_dim_state,))
    latent_series_vector = Input(shape=(latent_dim_series,))
    merge_layer = merge(
        [latent_state_vector, latent_series_vector], mode='concat')
    layer = Dense(np.prod(input_shape),
                  activation='sigmoid')(merge_layer)
    layer = Reshape(input_shape)(layer)
    return Model(input=[latent_state_vector, latent_series_vector], output=[layer, latent_series_vector])


def _ae_front_half(batch_size, n_steps, latent_dim_state, latent_dim_series):
    '''
    Returns a model that turns an input series of statematrices into a latent vector
    '''
    series = Input(batch_shape=(batch_size,
                                n_steps) + input_shape)
    layer = Reshape((n_steps, np.prod(
        input_shape)))(series)
    layer = TimeDistributed(Dense(latent_dim_state))(layer)
    layer = GRU(latent_dim_series)(layer)
    return Model(input=series, output=layer)
