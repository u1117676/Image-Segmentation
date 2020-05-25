# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:31:52 2020

@author: u1117676
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import IPython.display as display
import tensorflow.keras.applications.vgg16
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
pixels=350
#%%
#Importing my photos
Train_Image_Dir=pathlib.Path("ImageSets\Train_Dir")     # Images and labels for training
Valid_Image_Dir=pathlib.Path("ImageSets\Validation_Dir")     # Images and labels for training

image_count =len(list(Train_Image_Dir.glob('**/*.jpg')))
print(image_count)
#%%Image Processing

def Imagesetgen(trainpath, pixels):
    train_dir = trainpath

    # Add data-augmentation parameters to ImageDataGenerator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


    train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(pixels,pixels),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        )
  
    return train_generator
#%%
t_generator = Imagesetgen(Train_Image_Dir,350)