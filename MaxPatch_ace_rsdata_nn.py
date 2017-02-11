import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from ace_rsdata import *

X, Y = get_dataset('DX')
dummy_x = np.zeros(X[0].shape)
input_shape = np.expand_dims(np.moveaxis(dummy_x, -1, 0)[0], axis=-1).shape
print "Input shape: " + str(input_shape)

from toolkit.MaxPatch import *
import keras

print("Loading model")
model_file = "h5_files/ace_rsdata_nn_variant1_02_04.h5"
model = keras.models.load_model(model_file)


# Preprocess the data
print("Preprocessing the data ...")
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Decreasing input size to dramatically reduce training time for testing
X, Y = X[0:7], Y[0:7]

# Dividing into 5 categories because the oldest person is 19
Y = np_utils.to_categorical([int(y-1) for y in Y], nb_classes=4)

# X's shape is  ( , 96, 96, 50, 1) or (scans, x, y, z, dummy value)
files = X
X = [np.expand_dims(np.moveaxis(x.get_data(), -1, 0)[0], axis=-1) for x in X]
# X = [np.expand_dims(x.get_data(), axis=0) for x in X]

X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

def show3D(self):
    """Creates a matplotlib figure with different 'images' as columns, and their cross sections as rows."""
    if not self.patches:
        self.generate()

    fig, axarr = plt.subplots(self.patches[0].shape[-1]+2, len(self.patches))
    axarr, labels, locations = axarr[2:], axarr[0], axarr[1]
    
    fig.set_size_inches(len(self.patches), self.patches[0].shape[-1]+1)
    
    
    
    fig.suptitle("Patches corresponding to maximally active locations on layer: {:}, filter: {:}".format(self.layer.name, self.filter_number), y=0.0)
    for i in range(len(self.patches)):
        labels[i].axis('off')
        labels[i].text(0.0, 0.5,
          "{:}\nin {:}".format(self.max_locations[i], self.image_indices[i]), {"size": "smaller"} )
        
        # Show the maxlocation graphically
        sub = locations[i]
        max_loc = self.max_locations[i]
        image = X[self.image_indices[i]]
        print("image.shape", image.shape, 'max_loc', max_loc)
        sub.imshow(image[:,:,max_loc[2], 0], cmap='gray') # Image has shape 96,96,50,1
        sub.plot(max_loc[1],max_loc[0], 'go')
        sub.axes.get_xaxis().set_visible(False)
        sub.axes.get_yaxis().set_visible(False)
	# Abusing plot_connectome to show a single location
	# plot_connectome(np.array([[1]]), 
	#plotting.plot_anat(files[i])

        # Reshaping patch
        patch = self.patches[i]
        if len(patch.shape) == 4:
            patch = np.squeeze(patch, axis=(0,))
        if not len(patch.shape) == 3:
            raise ValueError("Expected patches to be 3D (or 4D with the first dimension of 1). Got shape {:}".format(self.patches[i].shape))

        # For each patch, draw all cross sections
        for z in range(patch.shape[-1]):
            subplot = axarr[z,i]
            subplot.axis('off')
            subplot.set_title('Z={:}'.format(z))
            subplot.imshow(patch[:, :, z], cmap="gray", interpolation='none')
    return fig




print("Making visualization")
convolutional_layers = MaxPatch.get_convolutional_layers(model)
images = list(map(lambda x:np.squeeze(x), X))
for layer in convolutional_layers:
    for i in range(layer.nb_filter):
        mp = MaxPatch(model, X, images=images, layer=layer, filter_number=i)
        mp.generate()
        print('Patch shape:', mp.patches[0].shape)
        show3D(mp).savefig('img/ace_rsdata_layer_'+layer.name+'_filter'+str(i)+'_max_patches.png')
        plt.close("all")
        #mp.save('img/ace_rsdata_layer_'+layer.name+'_filter'+str(i)+'_max_patches.png', dimensions=3)

