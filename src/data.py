import numpy as np
import pandas as pd
import copy as cp
import pdb
from itertools import product
from tqdm import tqdm
import io
import math
import random
import time
import multiprocessing

class Dataset(object):
    def __init__(self):
        self.different_pairs_size = None
        self.similar_pairs_size = None
        self.images_train = np.array([])
        self.labels_train = np.array([])
        self.images_test = np.array([])
        self.unique_train_label = np.array([])
        self.map_train_label = {}
        self.setup()

    def setup(self):
        start_time = time.time()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        diff_process = multiprocessing.Process(target=count_different_pairs, args=(0, return_dict))
        sim_process = multiprocessing.Process(target=count_similar_pairs, args=(1, return_dict))
        diff_process.start()
        sim_process.start()

        i = 0
        while diff_process.is_alive() or sim_process.is_alive():
            if i == 500000:
                print("Time is " + str(time.time() - start_time) + " since execution began")
                print("")
                i = 0
            i += 1

        sim_process.join()
        diff_process.join()
        self.different_pairs_size = return_dict[0]
        self.similar_pairs_size = return_dict[1]
        print("Setup time: " +  str(time.time() - start_time))
        # file_reader = pd.read_csv('datasets/train_pairs.csv', sep=',', header=True, index=False, chunksize=32)


    def get_next_batch(self):
        start_batch = time.time()
        similar_percent = np.random.rand()
        similar_num = int(32 * similar_percent)
        different_num = 32 - similar_num

        sort_time = time.time()
        skip_similar_rows = np.sort(np.random.randint(1, self.similar_pairs_size + 1,
            size=self.similar_pairs_size - similar_num))
        skip_different_rows = np.sort(np.random.randint(1, self.different_pairs_size + 1,
            size=self.different_pairs_size - different_num))
        print("Time for sort", time.time() - sort_time)

        read_similar = time.time()
        similar_batch = pd.read_csv('datasets/similar_pairs.csv', sep=',', header=None, skiprows=skip_similar_rows)
        print("Time for read similar", time.time() - read_similar)

        read_different = time.time()
        different_batch = pd.read_csv('datasets/different_pairs.csv', sep=',', header=None, skiprows=skip_different_rows)
        print("Time for read different", time.time() - read_different)

        print("Total batch time", time.time() - start_batch)
        print("Complete")
        pdb.set_trace()

        np.concatenate((similar_batch.values, different_batch.values), axis=1)


def remove_new_whale(dataset):
    dataset = dataset[dataset['Id'] != 'new_whale']
    dataset.to_csv('datasets/train_no_whale.csv', sep=',', header=True, index=False)

def split_dataset(dataset):
    dataset_size = dataset.shape[0]
    start_0, end_0 = 0, int(dataset_size * 0.25)
    start_1, end_1 = int(dataset_size * 0.25), int(dataset_size * 0.50)
    start_2, end_2 = int(dataset_size * 0.50), int(dataset_size * 0.75)
    start_3, end_3 = int(dataset_size * 0.75), int(dataset_size)
    dataset.iloc[start_0:end_0].to_csv('datasets/train_0.csv', sep=',', header=True, index=False)
    dataset.iloc[start_1:end_1].to_csv('datasets/train_1.csv', sep=',', header=True, index=False)
    dataset.iloc[start_2:end_2].to_csv('datasets/train_2.csv', sep=',', header=True, index=False)
    dataset.iloc[start_3:end_3].to_csv('datasets/train_3.csv', sep=',', header=True, index=False)

def create_pairs_dataset(dataset):
    values = dataset.values[:, 0]
    full_pairs = np.transpose([np.tile(values, values.shape[0]), np.repeat(values, values.shape[0])])
    del values
    return full_pairs

def split_pairs_dataset(dataset):
    similar_pairs = dataset[dataset['Label'] == True]
    similar_pairs.to_csv('datasets/similar_pairs.csv', sep=',', header=True, index=False)
    del similar_pairs
    different_pairs = dataset[dataset['Label'] == False]
    different_pairs.to_csv('datasets/different_pairs.csv', sep=',', header=True, index=False)

def count_different_pairs(i, return_dict):
    return_dict[i] = sum(1 for row in open('datasets/different_pairs.csv', 'r')) - 1

def count_similar_pairs(i, return_dict):
    return_dict[i] = sum(1 for row in open('datasets/similar_pairs.csv', 'r')) - 1

if __name__ == "__main__":
    dataset = Dataset()
    pdb.set_trace()
    print("Complete")

    # start_time = time.time()
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # diff_process = multiprocessing.Process(target=count_different_pairs, args=(0, return_dict))
    # sim_process = multiprocessing.Process(target=count_similar_pairs, args=(1, return_dict))
    # diff_process.start()
    # sim_process.start()
    #
    # i = 0
    # while diff_process.is_alive() or sim_process.is_alive():
    #     if i == 500000:
    #         print("Time is " + str(time.time() - start_time) + " since execution began")
    #         print("")
    #         i = 0
    #     i += 1
    #
    # sim_process.join()
    # diff_process.join()
    # pdb.set_trace()
    # print("Complete")
