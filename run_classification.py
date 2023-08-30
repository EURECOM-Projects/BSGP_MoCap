#! /usr/local/bin/ipython --

import sys
import os
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import numpy as np
import argparse
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from src.models import VariationalGaussianProcess
from src.utils import plot_precision_matrix, mocap_sensors_info
from torch.utils.data import  TensorDataset

import torch
from pprint import pprint

import seaborn as sns
import pandas as  pd
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

def save_results(args, model):
    plot_precision_matrix(path=args.results_path, 
                          filename='precision_matrix',
                          P=model.model.kernel.precision(), 
                          features=args.features)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def select_first_measure(x):
    N, T, D = x.shape
    d = 21
    x_stacked = np.vstack([x[i] for i in range(x.shape[0])])
    x_stacked_first_measure = np.empty((N*T, d))
    i = 0
    j = 0
    _, measures_per_sensor_50 = mocap_sensors_info('small')
    for sensor_n_measures in measures_per_sensor_50:
        x_stacked_sensor = x_stacked[:,i:i+sensor_n_measures] # NT x S
        x_ = x_stacked_sensor[:,0:1]
        x_stacked_first_measure[:,j:j+1] = x_
        j += 1
        i += sensor_n_measures
    x_ = np.concatenate([np.expand_dims(x_stacked_first_measure[i * T:(i + 1) * T], 0) for i in range(N)], 0)
    return x_

def create_mocap_dataset(subsampling=-1, features='full', static=False, fold=0):
    mocap09 = np.load('./data/mocap/mocap09.npz')['data']
    mocap07 = np.load('./data/mocap/mocap07.npz')['data']
    mocap08 = np.load('./data/mocap/mocap08.npz')['data']
    mocap16 = np.load('./data/mocap/mocap16.npz')['data']

    #  Sub-sampling of 1/8 [ss]
    if subsampling != -1:
        rate = subsampling
        mask = np.zeros(mocap09.shape[1], dtype=int)
        mask[::rate] = 1
        mocap09 = mocap09[:,mask==1,:]
        mask = np.zeros(mocap08.shape[1], dtype=int)
        mask[::rate] = 1
        mocap08 = mocap08[:,mask==1,:]
        mask = np.zeros(mocap07.shape[1], dtype=int)
        mask[::rate] = 1
        mocap07 = mocap07[:,mask==1,:]
        mask = np.zeros(mocap16.shape[1], dtype=int)
        mask[::rate] = 1
        mocap16 = mocap16[:,mask==1,:]

    # Level the second dimension (time)
    T = np.min([mocap09.shape[1], mocap07.shape[1], mocap08.shape[1], mocap16.shape[1]])
    mocap09 = mocap09[:,0:T,:]
    mocap07 = mocap07[:,0:T,:]
    mocap08 = mocap08[:,0:T,:]
    mocap16 = mocap16[:,0:T,:]
    walk = mocap09.shape[0] + mocap16.shape[0]
    run =  mocap07.shape[0] +  mocap08.shape[0]

    if features == 'small':
        mocap09 = select_first_measure(mocap09)
        mocap07 = select_first_measure(mocap07)
        mocap08 = select_first_measure(mocap08)
        mocap16 = select_first_measure(mocap16)

    labels = np.concatenate((np.ones(run), np.zeros(walk)))[:,None]
    Y = np.vstack((mocap09, mocap16, mocap07, mocap08))

    # Normalize
    data_std = Y[:, :].std((0, 1), keepdims=True) + 1e-5
    data_mean = Y[:, :].mean((0, 1), keepdims=True)
    Y = (Y - data_mean) / data_std

    # Treat zero readings
    if features == 'small':
        Y[:,:,(7,11)] = 1e-6
    else:
        Y[:,:,(24, 25, 31, 32)] = 1e-6

    # Reshape
    D = Y.shape[2]
    T = Y.shape[1]
    Y = Y.reshape(Y.shape[0],-1)

    # Split
    if static == False:
        return Y, labels, {'D': D, 'T': T}
    else:
        Y_train_indices_boolean = np.random.choice([1, 0], size=Y.shape[0], p=[0.8, 0.2])
        X_train_indices = np.where(Y_train_indices_boolean == 1)[0]
        X_test_indices = np.where(Y_train_indices_boolean == 0)[0]
        Y_train = Y[X_train_indices]
        Y_test = Y[X_test_indices]
        labels_train = labels[X_train_indices]
        labels_test = labels[X_test_indices]
        return Y_train, labels_train, Y_test, labels_test, {'D': D, 'T': T}

def main():
    set_seed(0)
    # Load data
    Y, labels, Yshape = create_mocap_dataset(subsampling=args.sub_sampling, features=args.features, static=False, fold=0)
    # Train model
    model = VariationalGaussianProcess({'Y':Y ,'labels': labels}, Yshape['D'], Yshape['T'], args.kernel_computation_type)
    model.train(args.max_iter)
    # Save results
    train_accuracy = model.compute_accuracy(Y, labels)
    print('Train Accuracy: %.2f%%'%(train_accuracy*100))
    save_results(args, model)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification experiment')
    parser.add_argument('--features', choices=['full','small'], default='full')
    parser.add_argument('--sub_sampling', type=int, default=-1)
    parser.add_argument('--results_path', type=str, default='./results')
    parser.add_argument('--kernel_computation_type', choices=['single_sum', 'double_sum'], default='single_sum')
    parser.add_argument('--max_iter', type=int, default=10000)



    args = parser.parse_args()

    main()