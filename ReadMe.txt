--- Description ---

We will develop a semantic segmentation U-net model to detect pixels with buildings. We will train the model with images where every pixel indicates whether it has a building/construction in it or not. We will test the model against images with unclassified images in order to predict its buildings identification accuracy.

Our date is contained in a Python pickle file called dataset.pickle. Both the training set and the validation set are lists. Each entry in the list is a dictionary with
keys “image” and “mask”. Both of these contain numpy arrays with image data.

The image data is captured from the European Space Agency’s (ESAs) Sentinel-2 satellite with pixel size of 10 m x 10 m, meaning that each image corresponds to 2.56 km x 2.56 km. The three channels correspond to raw channel readings (values in the range from 0 to thousands) from the sensor in Blue-Green-Red order. The mask is a binary mask with 1s denoting pixels that contain buildings and 0s elsewhere.

--- Instructions -----

Execute the following command in terminal in the current folder:

python segmentation_project.py  --dataset_name dataset.pickle --model_name Unet_model.h5

-----------------

I have python 3.9.5 in my laptop, but worked with the latest version on Google Collab. 3.9.5 is sufficient to install all the modules.

In the requirements.txt file I simply copied all the installed modules in my local python distribution, since it is recent and focused on machine learning.

In any case, most important modules should be keras, tensorflow, segmentation-models, pickle, albumentations, argparse, scikit-learn, numpy and pandas (in their latest versions).
