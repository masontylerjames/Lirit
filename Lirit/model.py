from features import n_features
from keras.layers import Input, Permute, TimeDistributed, LSTM, Dropout, Dense, Activation, Bidirectional
from keras.models import Model


def model(n_steps, dropout=0.5):
    '''
    The model architecture has two time-series LSTM layers that have shared weights over pitches
    This is followed by two bidirectional pitch-series LSTM layers.
    The final step is two dense layers with shared weights over pitches
    '''
    input_layer = Input(shape=(n_steps, 87, n_features))
    permute_1 = Permute((2, 1, 3))(input_layer)
    per_pitch_lstm_1 = TimeDistributed(
        LSTM(256, return_sequences=True), name='pitch_lstms_1')(permute_1)
    dropout_1 = Dropout(dropout)(per_pitch_lstm_1)
    per_pitch_lstm_2 = TimeDistributed(
        LSTM(256), name='pitch_lstms_2')(dropout_1)
    dropout_2 = Dropout(dropout)(per_pitch_lstm_2)
    pitch_lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(
        dropout_2)
    dropout_3 = Dropout(dropout)(pitch_lstm_1)
    pitch_lstm_2 = Bidirectional(
        LSTM(64, return_sequences=True))(dropout_3)
    dropout_4 = Dropout(dropout)(pitch_lstm_2)
    dense_1 = TimeDistributed(Dense(32))(dropout_4)
    dense_2 = TimeDistributed(Dense(2))(dense_1)
    out = Activation('sigmoid')(dense_2)
    model = Model(input=input_layer, output=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model
