# ERA1_S9_Assignment
ERA1 - Assignment for "Session 9 - Advanced Convolutions, Data Augmentation and Visualization "

Created by - Samir Paul

ERA1 - Assignment for "Session 9 - Advanced Convolutions, Data Augmentation and Visualization "

Dataset = CIFAR10 Used Pytorch Framework

## Objective
Write a new network that has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead)

(If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)

total RF must be more than 44

one of the layers must use Depthwise Separable Convolution

one of the layers must use Dilated Convolution

use GAP (compulsory):- add FC after GAP to target #of classes (optional)

use albumentation library and apply:

horizontal flip shiftScaleRotate coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None) achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## Work done and Results
one of the layers must use Dilated Convolution

Albumentation is used

Parameters = 164,488

Created following individual modules for transformations, models, training code and test code and called these modules from main Python Notebook code.


mymodels.py

mytrain.py

mytest.py

At Epoch = 71, Test Accuracy is 84.12%
