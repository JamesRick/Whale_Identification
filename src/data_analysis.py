import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from memory_profiler import profile
from PIL import Image, ImageFilter
from tqdm import tqdm
from skimage import color, data, restoration, io
from multiprocessing import Process

import pdb
import math
import time
import subprocess
import os
import gc
import sys

def class_number(train_dataset):
    return train_dataset['Id'].nunique()

def class_frequency(train_dataset):
    return train_dataset['Id'].value_counts()

def num_classes_freq_1(train_dataset):
    return train_dataset['Id'].value_counts()[train_dataset['Id'].value_counts() == 1].count()

def load_images_from_folder(folder_name, start, end, new_whale_list):
    images = []
    images_failed = []
    image_names = os.listdir(folder_name)
    for file_name in tqdm(image_names[start:end]):
        try:
            if str(file_name) not in new_whale_list:
                img = io.imread(os.path.join(folder_name, file_name))
            if img is None and not str(file_name) not in new_whale_list:
                raise ValueError()
            else:
                images.append(img)
        except:
            print("Path: " + str(os.path.join(folder_name, file_name)) + " failed to be loaded by imread")
            images_failed.append(str(os.path.join(folder_name, file_name)))
    # print("Size", sys.getsizeof(images))
    return images

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
def image_review(train_path, test_path):
    dataset = pd.read_csv('datasets/train.csv')
    new_whale_list = dataset[dataset['Id'] == 'new_whale']['Image'].values.tolist()

    train_shape_dict = {}
    test_shape_dict = {}
    train_size = os.listdir(train_path)
    start_0, end_0 = 0, int(len(train_size) * 0.25)
    start_1, end_1 = int(len(train_size) * 0.25), int(len(train_size) * 0.50)
    start_2, end_2 = int(len(train_size) * 0.50), int(len(train_size) * 0.75)
    start_3, end_3 = int(len(train_size) * 0.75), int(len(train_size))

    for batch in tqdm(range(4)):
        if batch == 0:
            train_images = load_images_from_folder(train_path, start_0, end_0, new_whale_list)
        elif batch == 1:
            train_images = load_images_from_folder(train_path, start_1, end_1, new_whale_list)
        elif batch == 2:
            train_images = load_images_from_folder(train_path, start_2, end_2, new_whale_list)
        elif batch == 3:
            train_images = load_images_from_folder(train_path, start_3, end_3, new_whale_list)

        for image in train_images:
            if image.shape in train_shape_dict.keys():
                train_shape_dict[image.shape] += 1
            else:
                train_shape_dict[image.shape] = 1
        del train_images

    test_size = os.listdir(test_path)
    start_0, end_0 = 0, int(len(test_size) * 0.25)
    start_1, end_1 = int(len(test_size) * 0.25), int(len(test_size) * 0.50)
    start_2, end_2 = int(len(test_size) * 0.50), int(len(test_size) * 0.75)
    start_3, end_3 = int(len(test_size) * 0.75), int(len(test_size))

    for batch in tqdm(range(4)):
        if batch == 0:
            test_images = load_images_from_folder(test_path, start_0, end_0, new_whale_list)
        elif batch == 1:
            test_images = load_images_from_folder(test_path, start_1, end_1, new_whale_list)
        elif batch == 2:
            test_images = load_images_from_folder(test_path, start_2, end_2, new_whale_list)
        elif batch == 3:
            test_images = load_images_from_folder(test_path, start_3, end_3, new_whale_list)

        for image in test_images:
            if image.shape in test_shape_dict.keys():
                test_shape_dict[image.shape] += 1
            else:
                test_shape_dict[image.shape] = 1
        del test_images

    pdb.set_trace()
    print("Trace Start")

    train_sorted = sorted(train_shape_dict.items(), key=lambda kv: kv[1])
    test_sorted = sorted(test_shape_dict.items(), key=lambda kv: kv[1])

    print("Train", train_sorted[-1], "Test", test_sorted[-1])
    print("Train", train_sorted[-2], "Test", test_sorted[-2])

# image = io.imread('datasets/train/00d4a4967.jpg')
# psf = np.ones((5,5)) / 25.0
# image = color.rgb2gray(image)
# old_image = Image.fromarray(np.uint8(image*255))
# image = restoration.richardson_lucy(image, psf, 10)
# new_image = Image.fromarray(np.uint8(image*255))
# old_image.show()
# new_image.show()

def train_reshape(train_path, train_dataset_path):
    dataset = pd.read_csv(train_dataset_path)
    new_whale_list = dataset[dataset['Id'] == 'new_whale']['Image'].values.tolist()
    train_images = os.listdir(train_path)
    for image in train_images:
        if image not in new_whale_list:
            with Image.open(os.path.join(train_path, image)) as img:
                new_img = img.resize((256, 256))
                new_img.save('datasets/images/train_256_256/' + str(image))

def test_reshape(test_path):
    test_images = os.listdir(test_path)
    for image in test_images:
        with Image.open(os.path.join(test_path, image)) as img:
            new_img = img.resize((256, 256))
            new_img.save('datasets/images/test_256_256/' + str(image))


if __name__ == '__main__':
    train_path = "datasets/images/train/"
    test_path = "datasets/images/test/"
    train_dataset_path = "datasets/train.csv"
    start_time = time.time()
    train_process = Process(target=train_reshape, args=(train_path, train_dataset_path))
    test_process = Process(target=test_reshape, args=(test_path,))
    train_process.start()
    test_process.start()
    i = 0
    while train_process.is_alive() or test_process.is_alive():
        if i == 50:
            print("Time is " + str(time.time() - start_time) + " since execution began")
            print("")
            i = 0
        i += 1
    test_process.join()
    train_process.join()
