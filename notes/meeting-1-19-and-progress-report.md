

### What we have
* Max patch
* (Convolutional) Auto encoder
* Network model

### What we want
* Saliency maps
* Deep dreams?
* Polished (& useful) documentation
* Working examples
* Auto encoder -> Regions of the brain/time sequences

### Week by week timeline:
1. Done :-(
2. By Sunday (1/22)
  - Max patch auto fitting finish
  - Done with Saliency maps + visualization
  - Documentation + a bit of paper work
3. By next Sunday (1/29)
  - Work on the network model
  - Analysis (accuracy, visualizations, etc.) of Auto encoder & other tools on the rsdata set
  - More documentation
4. By 2/5
  - Deep dreams? (1-2 weeks)
  - Some leeway time...
  - More projects
   - *Evaluate if we need to run into D term*
5. Just write the paper
6. Presentations



## Notes from the meeting
We're pretty sure that nilearn has a *simple* way to map brain images to the standard brain (`masker.fit_transform(img)`? - probably use other niftimasker function?, resampling?)

We'll need to make sure to use that before we run the `rsdata` dataset through standardization before working with it.

We're targeting finihsing up C-term, but it wouldn't be a disaster if we need to keep working into D-term.
We should think about deadlines for this a bit, but it shouldn't be an issue.
