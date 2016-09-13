# A *very* simple example neural network problem
# Where we try to approximate a line

# I used Jupyter notebook to run this, which allows matplotlib to be drawn...
# It takes a little while to install.
# You'll also need to uncomment the following line
# %matplotlib inline

import numpy as np
np.random.seed(123)

import matplotlib.pyplot as plt

import theano
from theano import tensor as T, function

from IPython import display
import time


noise_var = .5 # Noise - how difficult the problem is
n_samples = 1000 # Number of observations

n = np.random.normal(0, noise_var, n_samples)
x = np.arange(n_samples)
x = (x-x.mean()) / x.std()
w, b = np.random.randn(2)

d = w*x + b # pre-noise correct answer
y = d + n # noisy actual observations

# Visualization of input data
plt.plot(x, y, '.')
plt.plot(x, d, '-r', linewidth=2.)
plt.legend(('noisy observations', 'desired'))
plt.title('input-output map')

# Theano model
X, Y = T.vectors('input', 'desired')
W = theano.shared(.01)
B = theano.shared(0.)
Z = W*X + B
Cost = ((Y-Z)**2).mean() # cost is the mean squared value
params = [W, B] # Parameters that should be learned

# Doubling W in place to demonstrate how that works
updates = ((W, W*2), )
double_w = function([], [], updates=updates)


# Training the model
# cost = (d-y)**2
# updates:
# w = w-lr*grad(cost, w) = w + lr*2*(d-y)*x
# b = b-lr*grad(cost, b) = b + lr*2*d-y)
lr = .1
grads = [T.grad(Cost, p) for p in params]
updates = [(p, p - lr*g) for p,g in zip(params, grads)]

print(updates)


# Setup training function in Theano
train = function([X, Y], Cost, updates=updates)


# Train + visualization
epochs = 30
final_cost = []

for i in range(epochs):
    # for inp, out in zip(x, y)
    # this is all we need to train the model, to call train()
    final_cost.append(train(x, y))

    # some niceties
    what, bhat = W.get_value(), B.get_value()

    # Cost function
    plt.subplot(211)
    plt.cla()
    plt.title('cost: {}'.format(final_cost[-1]))
    plt.plot(final_cost, linewidth=2.)

    # parameter space
    plt.subplot(212)
    plt.plot(w, b, '*', linewidth=2.)
    plt.plot(what, bhat, '.r', linewidth=2.)
    plt.xlim([w-.5, w+.5])
    plt.ylim([b-.5, b+.5])
    plt.xlabel('w')
    plt.ylabel('b')

    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(1.0)



# Final results
plt.title('final results')
plt.plot(x,y, '.')
plt.plot(x,d, 'r', linewidth=2.)
plt.plot(x, what*x + bhat, 'c', linewidth=2.)
plt.legend(('noisy', 'desired', 'estimated'))
