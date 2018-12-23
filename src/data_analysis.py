import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from collections import Counter


from PIL import Image
from tqdm import tqdm

import pdb
import math
import time
import subprocess

def class_number(train_dataset):
    return train_dataset['Id'].nunique()

def class_frequency(train_dataset):
    return train_dataset['Id'].value_counts()

def num_classes_freq_1(train_dataset):
    return train_dataset['Id'].value_counts()[train_dataset['Id'].value_counts() == 1].count()

# Quick look at the classes in the train.csv
def classes_review(train_dataset):
    frequency = class_frequency(train_dataset)
    print("Images per class", frequency)
    print("Number of classes", class_number(train_dataset))
    print("Number of classes with only 1 image", num_classes_freq_1(train_dataset))
    frequency = frequency.drop(frequency[frequency.values == 1].keys())
    print("Images per class excluding classes with only 1 image", frequency)
    print("Number of classes excluding classes with only 1 image", frequency.keys().nunique())

# Review of different imagines in train and test folders
def image_review(train_dataset):
    new_whale_ids = train_dataset[train_dataset['Id'] == 'new_whale']['Image'].values

    with Image.open('datasets/train/' + str(id)) as image:
        pass

if __name__ == '__main__':
    train_dataset = pd.read_csv('datasets/train.csv')
    # classes_review(train_dataset)
    image_review(train_dataset)
