MQP B-term week 1

Three projects to do with neurocognitive data at
candi-store.umassmed.edu

MRI data, survey data to do with different types of brains and brain functionality available

start from resting state MRI

new dataset (previously used: public dataset with packages to process image data)
- everything discovered in this dataset is new!
- MUST KEEP SAFE, do not share outside of group; confidential data

possible to process/computer data/analysis on WPI's cluster server if there is an account
- running multiple GPUs
- ACE CLUSTER
	* apply for account
	* arcweb.wpi.edu/ace-cluster
- can indicate the number of servers and GPUs to use

Kong will try to figure out how to get us access or give us the data to use on the cluster
- until then, experiment with the public datasets
- example: search "ADHD dataset fmri"

Kong can also try to find the raw data he downloaded and upload it to the cluster for us

python packages for neural fmri data: nipy (has a bunny on it), nilearn
-package for nipy: nibabel
-example code available on the nilearn git.io page that we can reference and use

stanford course has an example of visual image capture per layer at:cs231n.stanford.edu

if this time/day works, stick with it (check with everybody for next week)

tools to use: freesurfer, matlab

freesurfer can analyze functional mri & diffusion, has multiple methods to process the data, but Kong's experience is mostly with matlab
- important for converting data formats, data correction (into templates)

also AAL (automated anatomical labeling) - software to partition the brain in different ways by different regions

Kong can ask his PhD student to talk to us about the (python) software he is using
- python can be connected to tensorflow and the convolutional network part of the implementation

as long as we are on the cluster, we can use different modules:
- caffe, tensorflow, matlab, etc.
- if there are any problems, message the admin and he will fix it

Kong will send us the page with the python packages and the ACE CLUSTER form page so we can start working on the cluster

Project Goals (12-week scope, leaves 1 for presentation):
1) get at least one dataset ready to implement the fMRI
2) extract the functional network by using new convolutional network we're making (or something else, if necessary)
3) new visualization
	+ first layer captures simple links, second layer captures more complicated structures, etc.
4) need to give raw signal as input (different from conventional model)

Breakdown:
- first 4 weeks spent on goal 1
- next 4 weeks spent on goal 2
- last 4 weeks spent on everything else

We can fill in the smaller details within the 4 weeks later in our team (non-advisor) meetings

Kong's goals:
- figure out how to download the private dataset from UMass and get it to us
- give us the public dataset in the meantime
- send us the 2 webpages to use as resources
