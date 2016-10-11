from keras import backend as K
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D
from keras.utils import np_utils
from bp0 import *
import pprint
import inspect
K.set_image_dim_ordering('th')

model, X_train = mnist_setup('example_model.h5')[:2]

l = 5
filt = 4
sal_model = Sequential()
pred_model = Sequential()
for layer in model.layers[:l]:
    sal_model.add(layer)
    pred_model.add(layer)
sal_model.add(BpZeroLayer(filt, model.layers[l].output_shape[1]))

def silly_loss(y_true, y_pred):
    return K.ones_like(y_true)

plt.figure(figsize=(10,10))
weights_before = sal_model.layers[0].get_weights()[0]
sal_model_snapshot = Sequential()
sal_model_snapshot.add(sal_model)
for image_in in range(9):
    sal_model.compile(loss='mse', optimizer='adadelta')
    sal_model.fit(np.array([X_train[image_in]]), pred_model.predict(np.array([X_train[image_in]])), batch_size=1, nb_epoch=3, show_accuracy=True, verbose=1,sample_weight=np.array([10]))
    filt_result = np.mean(np.squeeze(sal_model.layers[0].get_weights()[0] - weights_before), axis=0)
    filt_result /= np.linalg.norm(filt_result)
    result = convolve2d(np.squeeze(X_train[image_in]), filt_result)
    sal_model = sal_model_snapshot
    plt.subplot(3,3,image_in + 1)
    plt.imshow(result, cmap='gray')
    plt.xlabel("result of saliency map")
plt.show()
