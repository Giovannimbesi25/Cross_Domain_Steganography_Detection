# Cross_Domain_Steganography_Detection
The project focuses on the detection of steganography in digital images using neural networks. The repository includes implementations of the steganography algorithms and deep learning models used, together with results obtained through various experiments.

## Problem
The practice of concealing information within digital images poses a significant challenge in terms of digital security and privacy. Our project aims to help defend against such threats through the development of an advanced steganography detection system.

## What we have done
We conducted an in-depth study focused on steganography detection in digital images, employing a methodology centered around training various neural networks. Specifically, we utilized ResNet and EfficientNet models to train neural networks on the well-known [ALASKA2](https://www.kaggle.com/competitions/alaska2-image-steganalysis). dataset available on Kaggle . The primary objective was to obtain multiclass predictions for various types of steganography applied to images in the dataset.<br>
Subsequently, we expanded the scope of our study by introducing additional steganography algorithms, including LSB adaptive and S_UNIWARD. Using these algorithms, we generated our dataset containing steganographically altered images. To assess the effectiveness of the previously trained neural networks, we subjected the new dataset to the best-performing neural network identified in the initial training phase.


