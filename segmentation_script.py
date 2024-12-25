## Semantic Segmentation to identify buildings in satellite images ##
## 29 - 09 - 2024

######### MODULES ##################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models
import keras
import tensorflow as tf
import cv2

## For Data Augmentations

import albumentations as A
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
from scipy.ndimage import rotate

# In order to segmentation_models to work properly:
# %env SM_FRAMEWORK=tf.keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

keras.backend.set_image_data_format('channels_last')

#!pip install segmentation-models
import segmentation_models as sm

# from google.colab import drive
# drive.mount('/content/gdrive')
import argparse 
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from segmentation_models.losses import bce_jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score, recall, f1_score, f2_score

#############################################

#### DATA UPLOADING AND MANIPULATION ######

### Main inputs ####

#dataset_name = input("Please enter the dataset name: ")
#SM_model_name = input("Please enter the model name: ")

parser = argparse.ArgumentParser(description="Required external variables and files to compile semantic segmentation ")
parser.add_argument('--dataset_name', required=True, help='The name of the dataset file')
parser.add_argument('--model_name', required=True, help='The name of the deep learning model to use')
args = parser.parse_args()

dataset_name = args.dataset_name 
SM_model_name = args.model_name

with open(dataset_name,'rb') as f:
    data = pickle.load(f)
train_data = data['train']
TEST_data = data['val']

## Reestructuring training data:
train_images=[]
train_masks=[]

for i in range(len(train_data)):
    train_images.append(train_data[i]['image'])
    train_masks.append(train_data[i]['mask'])

## We convert lists to arrays for machine learning processes:
train_images = np.array(train_images)
train_masks = np.array(train_masks)

## Restructuring validation (actually test) data:
TEST_images=[]
TEST_masks=[]

for i in range(len(TEST_data)):
    TEST_images.append(TEST_data[i]['image'])
    TEST_masks.append(TEST_data[i]['mask'])

TEST_images = np.array(TEST_images)
TEST_masks = np.array(TEST_masks)


### DATA AUGMENTATION ####

## We will augment our data without the need to store it.

## Amount of images we want to generate in total:

images_to_generate=4000

train_images_augmented=[]
train_masks_augmented=[]

## We define the augmentation function based on albumentations module transformations:

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    ]
)

i=1   # variable to iterate till images_to_generate

while i<=images_to_generate:
    number = random.randint(0, len(train_images)-1)  #PIck a number to select an image & mask

    original_image = train_images[number]
    original_mask = train_masks[number]

    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

    train_images_augmented.append(transformed_image)
    train_masks_augmented.append(transformed_mask)

    i =i+1

train_images_augmented = np.array(train_images_augmented)
train_masks_augmented = np.array(train_masks_augmented)

print ('Data successfully augmented from 232 to 4000 images')

###################################

### TRAIN SPLIT FOR LATER FITTING
# Separate the test data. We separated based on the original augmented training set. We will train the model with this train and test data, to later evaluate it with the validation data provided in advance.

### TO TRAIN WITH AUGMENTED IMAGES:
x_train_0, x_val_0, y_train_0, y_val_0 = train_test_split(train_images_augmented, train_masks_augmented, test_size=0.20, shuffle=True)

##### PREPROCESSING ######

## PREPROCESSING ENCODER

#BACKBONE = 'resnet152'
#BACKBONE = 'resnet34'
#BACKBONE = 'vgg16'
BACKBONE = 'efficientnetb3'
#BACKBONE = 'vgg16'

## After trying several options, I decided to move on with efficientnetb3.
## We carry out the pre_processing based on efficientnet, a family of neural-network models.
## They offer pre-trained weights for the great database imagenet, of which we will make use later.

preprocess_input = sm.get_preprocessing(BACKBONE)


#Use customary x_train and y_train variables
x_train = x_train_0
y_train = y_train_0

y_train = np.expand_dims(y_train, axis=3) ## We match the sizes and data estructure in order for our code to compile later.
#print ('new Y train shape: ', np.shape(y_train))

x_val = x_val_0
y_val = y_val_0
y_val = np.expand_dims(y_val, axis=3) #May not be necessary.. leftover from previous code
#print ('new Y val shape: ', np.shape(y_val))

###
x_train= preprocess_input(x_train)
x_val= preprocess_input(x_val)

#####################################
##### SOME INFO ON THE SEMANTIC SEGMENTATION MODEL CREATION (UNEXECUTED IN THIS SCRIPT) #######

## After examining and practicing with Linknet, FPN and Unet, I decided to go on with Unet given its better convergence for the loss curves during training. Its an industry standard, and yielded the best results so far.
## As mentioned above, I choose the encoder weights for the database 'imagenet', given its popularity, size, compatibility and variety.


# model = sm.Unet(BACKBONE, encoder_weights='imagenet')
# model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall, sm.metrics.f1_score, sm.metrics.f2_score, 'mse'])


## The following command offers all the model characteristics.

# print(model.summary())

##### HOW I CARRIED OUT THE MODEL FIT / TRAINING (UNEXECUTED IN THIS SCRIPT) #####

 # history=model.fit(x_train,
 #         y_train,
 #         batch_size=16,
 #         epochs=150,
 #         verbose=1,
 #         validation_data=(x_val, y_val))

# model.save(...)

###########################
##### MODEL LOADING #######

##################
## Characteristics:
###################

##### Model: U-net
##### Backbone: Efficientnetb3
##### Weights/Dataset: Imaginet
##### Number of total (augmented) data: 4000
##### Train / Split : 80 / 20 [%] (with shuffle)
##### Augmentations: VerticalFlip, HorizontalFlip, Transpose, RandomRotate90 (all with probability of 0.5 to occur)
##### Compiler: Adam
##### Number of epochs: 150
##### Batch Size (for training): 16


TEST_images_preproc= preprocess_input(TEST_images)
TEST_masks_f = np.expand_dims(TEST_masks, axis=3)


#model_aug4000_NOTnorm_150epochs_16batchsize_Unet_imagenet_efficinetb3_TRAINSPLIT.h5

model=keras.models.load_model(SM_model_name, compile=False)
model.compile(loss=[bce_jaccard_loss], metrics=[iou_score, recall, f1_score, f2_score])

## Here we evaluate the model and show on screen the main metrics:

accuracy = model.evaluate(TEST_images_preproc, TEST_masks_f, batch_size=32)

# print(f"Test Loss: {loss}") 
print(f"Test Accuracy: {accuracy}")


