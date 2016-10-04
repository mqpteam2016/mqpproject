# Here's a simple Keras CNN.
# Again, I ran it using jupyter notebook, so it may take a bit of work to replicate
# This is from Eder Santana's tutorials

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D
from keras.utils import np_utils
from keras.backend.common import _FLOATX
import pprint
import inspect
K.set_image_dim_ordering('th')


class BpZeroLayer(Convolution2D):
    '''
    Make a 2d convolution that blacks out all but one filter
    '''
    def __init__(self, filter_index, input_filters):
        super(BpZeroLayer, self).__init__(16, 1,  1, trainable=False, border_mode='same', weights=[np.array([ [ [[1]] if i == filter_index else [[0]] for i in range(0,input_filters) ] for j in range(0, input_filters)]), np.zeros(input_filters)])

class BpOneLayer(Convolution2D):
    '''
    Make a 2d convolution that can be swapped for a BpZeroLayer
    '''
    def __init__(self, input_filters):
        super(BpOneLayer, self).__init__(input_filters, 1, 1, trainable=False, init='one')
ex_model = load_model('example_model.h5')

batch_size = 128
nb_classes = 10  # 10 digits from 0 to 9

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# for every filter
X_max = []
for k in list(range(16)):
    print("We're on filter:", k)
    model = Sequential()
    inp = 4
    for i in range(0,inp):
        model.add(ex_model.layers[i])
    model.add(BpZeroLayer(k, 16))
    print("in the middle:")
    print(len(model.layers[inp].get_weights()[0]))
    print(len(model.layers[inp].get_weights()[0][0]))
    print(len(model.layers[inp].get_weights()[0][0][0]))
    print(len(model.layers[inp].get_weights()[0][0][0][0]))
    for i in range(inp,  len(ex_model.layers)):
        model.add(ex_model.layers[i])
    #model.layers[inp].set_weights(BpZeroLayer(i, 16).get_weights())
    # for every piece of data
    min_discrepancy = float("inf")
    X_max.append(0)
    for j in list(range(400)):#X_train.shape[0])):
        print(j)
        disc = np.dot(model.predict_on_batch(np.array([X_train[j]])), Y_train[j])
        print(disc[0])
        if min_discrepancy > disc[0]:
            min_discrepancy = disc[0]
            X_max[k] = j
    del model
print(X_max)
