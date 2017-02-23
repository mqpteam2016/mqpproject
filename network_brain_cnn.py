from visualize_brain import *
from keras import backend as K
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D
from keras.layers import merge, Lambda
from keras.utils import np_utils
from bp0 import *
from pprint import pprint
import inspect
K.set_image_dim_ordering('th')
from ace_rsdata import *
from keras_gradients import *
#from deconv3d import *
from keras.layers.convolutional import Convolution3D
from unpooling import Unpooling3D

from nilearn import plotting
import nibabel

import pickle

ds = get_dataset()

itemlist = None
with open('outfile', 'rb') as fp:
    itemlist = pickle.load(fp)
items = []
for i in itemlist:
    items = np.array(i)
itemedges = []
for i in range(len(items)):
    itemedges.append([])
    for j in range(items[i].shape[0]):
        for k in range(j, items[i].shape[0]):
            it = items[i]
            itemedges[i].append(np.array([it[j], it[k]]))
    itemedges[i] = np.array(itemedges[i])

model = Sequential()
model.add(Convolution2D(4, 2, 1, input_shape=(1,2,156), border_mode='valid'))
model.add(Convolution2D(3, 1, 6, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Convolution2D(3, 1, 6, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(1,3)))
model.add(Convolution2D(1, 1, 23, border_mode='valid'))
edges = [model for i in range(items.shape[1]*int((items.shape[1]-1)/2))]
merged_vector = Lambda(lambda x : merge(x, mode='concat', concat_axis=-1))([e for e in itemedges])
predictions = Dense(1, activation='sigmoid')(merged_vector)

predictions.compile(loss='categorical_crossentropy', optimizer='rmsprop')
nb_epoch=1
predictions.fit(np.array(itemedges), ds_Y = list(map(lambda x: x.values[0], ds[1][:1])), nb_epoch=nb_epoch, batch_size=1)
