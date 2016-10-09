# Here's a simple Keras CNN.
# Again, I ran it using jupyter notebook, so it may take a bit of work to replicate
# This is from Eder Santana's tutorials

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from theano import tensor as T
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, Deconvolution2D
from keras.utils import np_utils
from keras.backend.common import _FLOATX
import pprint
import inspect
K.set_image_dim_ordering('th')
from bp0 import *


class Unpooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(Unpooling2D,self).__init__()
        #self.input = T.tensor4()
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

ex_model, X_train, Y_train = mnist_setup('example_model.h5')[:3]
models = []

for i in range(0, len(ex_model.layers)):
    l = ex_model.layers[i]
    model2 = Sequential()
    if isinstance(l, Activation):
        j = i - 1
        l2 = ex_model.layers[j]
        while not isinstance(l2, Activation):
            if isinstance(l2, Convolution2D):
                print([l2.nb_filter, l2.nb_row, l2.nb_col, l2.input_shape, l2.output_shape[-3:]])
                model2.add(Deconvolution2D(nb_filter=l2.nb_filter,
                                           nb_row=l2.nb_row,
                                           nb_col=l2.nb_col,
                                           output_shape=(l2.input_shape[-3:]),
                                           input_shape=(l2.output_shape[-3:])))
            elif isinstance(l2, MaxPooling2D):
                model2.add(Unpooling2D(l2.pool_size))
            j = j - 1
            l2 = ex_model.layers[j]
        models.append(model2)
        del model2
            
# from here what we want to do is read from the pickle that we make using activation_data
# and run train our deconvolution models in models
# using an l2 loss function
