import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
from string import punctuation


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # reshape each 
    # X = np.asarray(X)
    # X.shape = (np.shape(X)[0:2])
    # y = np.asarray(y)
    # y.shape = (len(y), 1)

    # print(X.shape)
    # print(y.shape)

    for i in range(0, len(series) - window_size - 1):
        # x_ser = np.array(series[i:i + window_size])
        # x_ser.shape = X.shape
        # X = np.append(X, x_ser, axis=0)
        # y = np.append(y, series[i + window_size + 1], axis=0)



        X.append(list(series[i:i + window_size]))
        y.append([series[i + window_size + 1]])

    return np.array(X), np.array(y)
    # return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()

    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    model.summary()

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation_excep = ['!', ',', '.', ':', ';', '?']

    text_new = text.lower()
    text_new = ''.join([c for c in text_new if c not in punctuation or c in punctuation_excep])

    return text_new

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs

    inputs = []
    outputs = []

    for i in range(0, len(text) - window_size - 1, step_size):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size + 1])

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))

    model.summary()

    return model

