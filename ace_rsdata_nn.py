######
# Simple Neural Net for use on the ace cluster
######
# A simple neural network that predicts Age when given a raw slice of MRI data



# Get the data
import numpy as np
from ace_rsdata import get_dataset

X, Y = get_dataset('Age')
input_shape = X[0].shape
print "Input shape: " + str(input_shape)

# Assemble the NN
new_model = True
model_file = 'ace_rsdata_nn.h5'
model = None
if new_model:
	from keras.models import Sequential
	from keras.layers.core import Dense, Activation, Flatten, Reshape
	from keras.layers.convolutional import Convolution3D, MaxPooling3D, AveragePooling3D
	from keras.utils import np_utils
	from keras import backend as K
	K.set_image_dim_ordering('tf')

	model = Sequential()

	model.add(AveragePooling3D(input_shape=input_shape, pool_size=(10, 10, 10)))
	model.add(Convolution3D(16, 2, 5, 5, border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=(5, 5, 5)))
	shape = model.layers[-1].output_shape
	model.add(Flatten())
	#model.add(Reshape((shape[1]*shape[2]*shape[3]*shape[4],)))
	model.add(Dense(5))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')
else:
	import keras
	model = keras.models.load_model(model_file)


# Preprocess the data
print("Preprocessing the data ...")
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Dividing into 5 categories because the oldest person is 19
Y = np_utils.to_categorical([int(y/4) for y in Y], nb_classes=5)

X = [np.expand_dims(x.get_data(), axis=0) for x in X]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)



# Train the NN and periodically save it
print("Starting training")
for i in range(len(X_train)*5):
	current_index = i % len(X_train)
	model.train_on_batch([X_train[current_index]], [np.array([y_train[current_index]])])
	print('Trained epoch ' + i)
	if i % 2:
		try:
			os.rename(model_file, model_file + '_epoch_' + str(i-2))
		except OSError, e:
			print(model_file + " Doesn't yet exist")
		model.save(model_file)
