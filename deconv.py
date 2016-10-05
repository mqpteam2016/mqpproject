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
from keras.layers.core import Dense, Dropout, Activation, Flatten, fork
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, Deconvolution2D
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


class Unpooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(Unpooling2D,self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize,
            "ignore_border":self.ignore_border}
    
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
    for i in range(0, len(ex_model.layers)):
        l = ex_model.layers[i]
        model.add(l)
        if isInstance(l, Convolution2D):
            model2.add(Deconvolution2D(l.nb_filter, l.nb_row, l.nb_col, input_shape=(l.output_shape), 
        else if isInstance(l, Pooling2D):
            model2.add(Unpooling2D(l.pool_size))
    min_discrepancy = [float("inf") for i in range(0,9)]
    X_max.append([0 for i in range(0,9)])
    for j in list(range(400)):#X_train.shape[0])):
        print(j)
        disc = np.dot(model.predict_on_batch(np.array([X_train[j]])), Y_train[j])
        print(disc[0])
        if max(min_discrepancy) > disc[0]:
            min_discrepancy[np.argmax(min_discrepancy)] = disc[0]
            X_max[k][np.argmax(min_discrepancy)] = j
    del model
print(X_max)
