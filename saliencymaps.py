from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D
from keras.utils import np_utils
import * from bp0
import pprint
import inspect
K.set_image_dim_ordering('th')

model, X_train = mnist_setup('example_model.h5')[:2]

l = 10
filt = 2
function_layer = K.function([model.layers[0].input, 0], [model.layers[l].output])

vis_layer = BpZeroLayer(filt, model.layer.get_output_at(0))

K.function([vis_layer.input, 0], [vis_layer.output])
grads = K.gradients(loss, input_img)[0]