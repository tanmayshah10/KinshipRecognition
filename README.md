# KinshipRecognition
#### Kinship Recognition Kaggle Competition for COMS W4995: Topics in Computer Science - Deep Learning

This repository implements an ensemble of Siamese Networks to predict whether 2 individuals are of the same kin.

### Directory Structure

#### Requires creation of folders to store results and model weights.

#### AugmentTrainData.ipyb
Augments training data with more samples. Allows user to set the ratio of 0/1 samples in the dataset.
#### VGGEnsemble.ipynb
Main notebook to run the model implementation.
#### utils
Contains auxiliary functions, classes for the main model, as well as RBF and Cross-Attention layers.

    .
    |-- AugmentTrainData.ipynb
    |-- .gitignore
    |-- README.md
    |-- utils
    |   |-- funcs.py
    |   |-- layers.py
    |   |-- models.py
    |   |-- tf2_keras_vggface
    |       |-- __init__.py
    |       |-- models.py
    |       |-- README.md
    |       |-- utils.py
    |       |-- version.py
    |       |-- vggface.py
    |-- VGGEnsemble.ipynb

    2 directories, 13 files
