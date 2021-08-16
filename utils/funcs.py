from collections import defaultdict
from glob import glob
from random import sample
import cv2
import numpy as np
import pandas as pd


def get_train_val(family_name, train_folders_path, train_file_path):
    
    '''Gets training and validation sets.'''
    
    # Code of validation families beginning F0XXX
    val_families = family_name
    
    # List all images and get train and val sets.
    all_images = glob(train_folders_path + "*/*/*.jpg")
    train_images = [x for x in all_images if val_families not in x]
    val_images = [x for x in all_images if val_families in x]

    train_person_to_images_map = defaultdict(list)
    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]
    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)
    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
    
    # Get image pairs and labels from training data.
    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values, relationships.relationship.values))
    relationships = [(x[0],x[1],x[2]) for x in relationships if x[0][:10] in ppl and x[1][:10] in ppl]    
    
    # Return labelled training and validation sets.
    train = [x for x in relationships if val_families not in x[0]]
    val = [x for x in relationships if val_families in x[0]]
    return train, val, train_person_to_images_map, val_person_to_images_map


def read_img(path, input_shape):
    
    '''Reads and preprocesses images'''
    
    # Read image
    img = cv2.imread(path, -1)
    # Resize image to input shape
    img = cv2.resize(img, input_shape)
    # Normalize pixel values
    img = cv2.normalize(img,  np.zeros(img.shape[:2]), 0, 255, cv2.NORM_MINMAX)
    return np.array(img).astype(np.float)


def gen(list_tuples, person_to_images_map, input_shape, train_folders_path, batch_size=16, normalization='base'):
    
    '''Generator to feed training data.'''
    
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size)
        
        # All the samples are taken from train_ds.csv, labels are in the labels column
        labels = []
        for tup in batch_tuples:
            labels.append(tup[2])
        labels = np.array(labels)

        # Original images preprocessed
        X1 = [x[0] for x in batch_tuples]
        X1 = np.array([read_img(train_folders_path + x, input_shape) for x in X1])
        X2 = [x[1] for x in batch_tuples]
        X2 = np.array([read_img(train_folders_path + x, input_shape) for x in X2])
        
        # Mirrored images
        X1_mirror = np.asarray([cv2.flip(x, 1) for x in X1])
        X2_mirror = np.asarray([cv2.flip(x, 1) for x in X2])
        X1 = np.r_[X1, X1_mirror]
        X2 = np.r_[X2, X2_mirror]
        
        yield [X1, X2], np.r_[labels, labels]

        
def ignore_layers(model_name):
    
    '''Number of layers to ignore from top and bottom of backbone'''
    
    if model_name == 'senet50':
        ignore_bottom = -6
    elif model_name == 'resnet50':
        ignore_bottom = -5 
    elif model_name == 'vgg16':
        ignore_bottom = -2
    else:
        raise ValueError("model_name not recognized. Please choose one of 'vgg16', 'resnet50', 'senet50'.")
    ignore_top = 0
    return ignore_bottom, ignore_top
    