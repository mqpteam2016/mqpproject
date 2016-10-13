
# coding: utf-8

# In[9]:

get_ipython().magic('matplotlib inline')
# Allows us to create Deep Dream style visualizations

# Heavily inspired by https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
from keras.models import load_model
from keras import backend as K
from max_patch import get_convolutional_layers
import numpy as np


# In[80]:

#### COPIED FROM https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html


def get_image(layer_name='conv2/3x3_reduce', filter_index=0, step=lambda i: 0.1, iterations=50):

    colors, img_width, img_height = model.layers[0].input_shape[-3:]

    input_img = model.layers[0].input

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = -K.mean(layer_output[:, filter_index, :, :]) +K.mean(layer_output) # We've been playing around with this

    # compute the gradient of the input picture with this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = np.random.random((1, colors, img_width, img_height)) * 100 + 128.
    # run gradient ascent for 20 steps
    
    import tqdm
    
    for i in tqdm.tqdm(range(iterations)):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step(i)

    return input_img_data



# In[81]:


#model = load_model('example_model.h5')
from googlenet.googlenet import *

model = create_googlenet('googlenet/googlenet_weights.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_dict


# In[ ]:

for i in range(16):
    import matplotlib.pyplot as plt
    plt.figure()
    
    image = get_image(layer_name='inception_5b/5x5', filter_index=i, step=lambda i: max(500/(i+1),0.05))[0]
    # Unpreprocess the image
    image = image.transpose(1,2,0)
    image += 115
    
    plt.imshow(image[:,:,0])
    plt.show()


# In[72]:

# Figuring out how to undo a transposition
shape = (2,3,5)
import numpy as np
A = np.array([ [ [k+j*3+i*9 for k in range(shape[2])] for j in range(shape[1])] for i in range(shape[0])])
T = A.transpose(2,0,1)

X = T.transpose(1,2,0)
print(X == A)

from scipy.misc import imread, imresize

img = imresize(imread('googlenet/cat_pictures/cat1.jpg', mode='RGB'), (224, 224)).astype(np.float32)
plt.imshow(imresize(imread('googlenet/cat_pictures/cat1.jpg', mode='RGB'), (224, 224)).astype(np.uint8))

plt.figure()
img[:, :, 0] -= 123.68
img[:, :, 1] -= 116.779
img[:, :, 2] -= 103.939
img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
img = img.transpose((2, 0, 1))
#img = np.expand_dims(img, axis=0)


image = img.transpose(1,2,0)
#image += 115
print(image.shape)
image[:,:,[0,0,0]] = image[:,:,[2,1,0]]
img[:, :, 0] += 123.68
img[:, :, 1] += 116.779
img[:, :, 2] += 103.939
plt.imshow(image)
print(image.shape)
image[:2,:2,:]


# In[67]:

imresize(imread('googlenet/cat_pictures/cat1.jpg', mode='RGB'), (224, 224))


# In[ ]:



