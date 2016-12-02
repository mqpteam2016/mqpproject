## Meeting December 1

### What did we do
* Got Convolutional AutoEncoder for 3D images to work on Ryan's computer
    * A step to get a variational autoencoder - allows us to figure out what an exagerrated brain in a category looks like
    * Keras doesn't have built-in 4D layers
    * We should train in steps (first layer to undo first layer, etc.)
* Shuffled around architecture for the toolkit
* The basic NN on the new RSData predicts patient's age.
    * Likely leads to overfitting, we should do initial training using an autoencoder type situation.
* Worked on the paper
    * Decided on some of the techniques we're putting into the toolkit (MaxPatch, Deconv, Saliency Maps)

It would be cool to have some video visualizations.

### What we will do

* Work on Convolutional AutoEncoder
    * Port to ACE
    * Train layer by layer
* Continue working on the paper

* Create dummy example for toolkit/MaxPatch
