from __future__ import print_function

# This is due to issues on the ace cluster
import sys
sys.path.append("/usr/lib64/python2.7/site-packages")
sys.path.append("/home/mgaskell/.local/lib/python2.7/site-packages")
sys.path.append("/home/mgaskell/work/mqpproject/ENV/lib64/python2.7")

sys.path.insert(0, '../toolkit')
sys.path.insert(0, '../')
import matplotlib as mpl
mpl.use('Agg')


from keras.models import load_model
from keras.datasets import mnist
import numpy as np

from ace_rsdata import get_dataset
from MaxPatch import MaxPatch

print("Loading model")
ex_model = load_model('../ace_rsdata_nn_12_5.h5')

print("Loading dataset")
X, Y = get_dataset('Age')
# X, Y = X[0:1], Y[0:1]
X = [np.expand_dims(np.moveaxis(x.get_data(), -1, 0)[0], axis=-1) for x in X]

print("Input shape: ", X[0].shape)

# Draw filter 0 from the last convolutional layer
layer = MaxPatch.get_convolutional_layers(ex_model)[-1]
filter_num = 0


print("Generating max patch")
viz = MaxPatch(ex_model, X, patch_size=(8,8,8,1), layer=layer, filter_number=filter_num)
viz.generate()

print("There are {:} patches which have shape: {:}".format(len(viz.patches), viz.patches[0].shape))

patches = [np.reshape(patch, (8,8,8)) for patch in viz.patches]

import pickle
with open('patches.pickle', 'w') as f:
	pickle.dump(patches, f)

# Now let's draw the patches:
print("Drawing max patch")
try:
	import matplotlib.pyplot as plt

	layer_name = layer.name

        for j in range(len(patches)):
            for i in range(patches[j].shape[0]):
                idx = (len(patches)-1)*j+i+1
                plt.subplot(len(patches), patches[j].shape[0], idx)
                if i== 0:
                    plt.ylabel('P{:}'.format(j+1))
                if j == len(patches)-1:
		    plt.xlabel("Z = {:}".format(i))
                plt.gca().get_xaxis().set_ticks([])
                plt.gca().get_yaxis().set_ticks([])
                plt.imshow(patches[j][i], cmap='gray', interpolation="none")
            if j == 0:
                plt.title('Max patch viz patch, {:}, filter: {:}'.format(layer_name, filter_num))



	plt.title('Max patch viz patch, {:}, filter: {:}'.format(layer_name, filter_num))

	plt.savefig('MaxPatch_{:}_{:}.png'.format(layer_name, filter_num))
except ImportError, e:
	print('Error creating an image of the visualization')
	print(e)
	print("Because there's been an error, we'll print out Patch[0]:")
	print(patches[0])
