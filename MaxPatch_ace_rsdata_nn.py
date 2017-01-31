import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from ace_rsdata import *
import os

X, Y = get_dataset('Age')
dummy_x = np.zeros(X[0].shape)
input_shape = np.expand_dims(np.moveaxis(dummy_x, -1, 0)[0], axis=-1).shape
print "Input shape: " + str(input_shape)

from toolkit.MaxPatch import *
import keras

print("Loading model")
model_file = "h5_files/ace_rsdata_nn_01_28.h5"
model = keras.models.load_model(model_file)


# Preprocess the data
print("Preprocessing the data ...")
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Decreasing input size to dramatically reduce training time for testing
X, Y = X[0:7], Y[0:7]

# Dividing into 5 categories because the oldest person is 19
Y = np_utils.to_categorical([int(y/4) for y in Y], nb_classes=5)

# X's shape is  ( , 96, 96, 50, 1) or (scans, x, y, z, dummy value)
X = [np.expand_dims(np.moveaxis(x.get_data(), -1, 0)[0], axis=-1) for x in X]
# X = [np.expand_dims(x.get_data(), axis=0) for x in X]

X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


print("Making visualization")
convolutional_layers = MaxPatch.get_convolutional_layers(model)
images = list(map(lambda x:np.squeeze(x), X_train))
for layer in convolutional_layers:
    for i in range(layer.nb_filter):
         mp = MaxPatch(model, X_train, images=images, layer=layer, filter_number=i)
         mp.generate()
         print('Patch shape:', mp.patches[0].shape)
         mp.save('img/ace_rsdata_layer_'+layer.name+'_filter'+str(i)+'_max_patches.png', dimensions=3)

