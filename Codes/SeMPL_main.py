# group according to scott knott test of training samples, instead of all samples
import numpy as np
from sklearn.metrics import silhouette_score
from numpy import genfromtxt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import time
import argparse
import random
from random import sample
import tensorflow as tf
import os
import copy
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

enable_tf2 = True
if enable_tf2:
    from Meta_sparse_model_tf2 import MTLSparseModel
    from mlp_sparse_model_tf2 import MLPSparseModel
    from mlp_plain_model_tf2 import MLPPlainModel
    from Meta_plain_model_tf2 import MTLPlainModel
else:
    from Meta_sparse_model import MTLSparseModel
    from mlp_sparse_model import MLPSparseModel
    from mlp_plain_model import MLPPlainModel
    from Meta_plain_model import MTLPlainModel

# Main function
if __name__ == '__main__':
    print('Read whole dataset from csv file ...')
    # dir_data = 'Data/deeparch-SizeReduction-3tasks.csv'
    # dir_data = 'Data/sqlite-overwritebatch-4tasks.csv'
    # dir_data = 'Data/sac_srad_others-5tasks.csv' #total_tasks = 5
    # dir_data = 'Data/gcc-diff_input-5tasks.csv'
    # dir_data = 'Data/spear-10286-6tasks.csv'  # total_tasks = 6
    # dir_data = 'Data/storm-obj2-3tasks.csv'
    # dir_data = 'Data/imagemagick-4tasks.csv'
    # dir_data = 'Data/xz-5tasks.csv'
    # dir_data = 'Data/x264-diff_input-10tasks.csv'  # total_tasks = 10

    dir_datas = ['Data/deeparch-SizeReduction-3tasks.csv', 'Data/sqlite-overwritebatch-4tasks.csv',
                 'Data/sac_srad_others-5tasks.csv', 'Data/gcc-diff_input-5tasks.csv', 'Data/spear-10286-6tasks.csv',
                 'Data/storm-obj2-3tasks.csv', 'Data/imagemagick-4tasks.csv', 'Data/xz-5tasks.csv',
                 'Data/x264-diff_input-10tasks.csv']
    selected_dir = [8]
    for dir, dir_data in enumerate(dir_datas):
        if dir in selected_dir:
            total_tasks = int(dir_data.split('tasks')[0].split('-')[-1])
            print('Dataset: ' + dir_data)
            whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
            (N, n) = whole_data.shape
            n = n - 1

            # delete the zero-performance samples
            non_zero_indexes = []
            delete_index = set()
            temp_index = list(range(N))
            for i in range(total_tasks):
                temp_Y = whole_data[:, n - i]
                for j in range(len(temp_Y)):
                    if temp_Y[j] == 0:
                        delete_index.add(j)
                # temp_Y = np.delete(temp_Y,[delete_index],axis=0)
            non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))

            save_file = True
            test_mode = False
            read_meta_model = True
            learning_model = 'deepperf'
            seed = 2
            print('Total sample size: ', N)
            N_features = n + 1 - total_tasks
            print('Number of features: ', N_features)
            N_train = 1 * N_features
            # N_train = 24
            N_meta_tasks = total_tasks-1
            # storm[158,261,422,678,903], imagemagick[11, 24, 45, 66, 79], xz[4, 8, 12, 16, 20], x264[24, 53, 81, 122, 161]

            meta_samples = int(0.75*N)

            learned_tasks = [0,1,2,3,4,5,6,7,9]
            learned_meta_models = ['']

            if meta_samples > len(non_zero_indexes):
                # N_test = int(len(non_zero_indexes) * 3 / 10)
                meta_samples = (len(non_zero_indexes))
            if N_train > N * 7 / 10:
                N_test = int(len(non_zero_indexes) * 3 / 10)
                N_train = (len(non_zero_indexes) - N_test)
            elif (N_train > 2):
                # N_test = int(N_train * 3 / 7)
                N_test = (len(non_zero_indexes) - N_train)
                # N_test = int(N * 3 / 10)
            else:
                N_test = 1

            saving_best_meta = 'best_meta_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed, meta_samples)
            print('Reading best meta models from {}'.format(saving_best_meta))

            meta_to_train = {}
            with open(saving_best_meta, 'r') as f:  # save the results
                lines = f.readlines()[0]
                import ast

                meta_to_train = ast.literal_eval(lines)
            print(meta_to_train)

            print('Meta sample size: {}'.format(meta_samples))
            print("Training sample size: {}".format(N_train))
            print('Testing sample size: ', N_test)

            N_experiments = 30
            start = 0
            print('Number of expriments: ', N_experiments)

            binary_system = True

            if np.max(whole_data[:, 0:N_features]) > 1:
                binary_system = False
                print('Numerial system')
            else:
                print('Bianry system')

            rel_error_mean = []
            rel_error_mean2 = []
            lambda_all = []
            error_min_all = []
            rel_error_min_all = []
            n_layer_all = []
            lr_all = []
            abs_error_layer_lr_all = []
            time_all = []

            for main_task in meta_to_train:
                for meta_tasks in meta_to_train[main_task]:
                    meta_tasks = meta_tasks[(total_tasks - N_meta_tasks - 1):]
                    if int(main_task) not in learned_tasks and meta_tasks not in learned_meta_models:
                        saving_file_name = 'Meta_{}_main{}_meta{}_{}_{}-{}_{}.txt'.format(
                            dir_data.split('/')[1].split('.')[0],
                            main_task, meta_tasks, seed,
                            N_train,
                            meta_samples,
                            time.strftime('%m-%d_%H-%M-%S',
                                          time.localtime(
                                              time.time())))
                        print('saving_file_name: {}'.format(saving_file_name))
                        main_task = int(main_task)
                        if save_file:
                            with open(saving_file_name, 'w') as f:  # save the results
                                f.write('N_train={} N_test={} '.format(N_train, N_test))

                        print('---Training main task {} with meta model {}---'.format(main_task, meta_tasks))

                        for ne in range(start, start + N_experiments):
                            print('\nExperiment {}: '.format(ne + 1))

                            results_deepperf = []
                            results_mtl = []
                            time_mtl = []
                            time_deepperf = []

                            if save_file:
                                with open(saving_file_name, 'a') as f:  # save the results
                                    f.write('\nRun {}'.format(ne + 1))

                            random.seed(ne * seed)

                            # Start measure time
                            start_time = time.time()
                            # generating training and testing data
                            training_index = []
                            testing_index = []
                            data_all = []
                            # feature_data = []

                            # delete the zero-performance samples
                            non_zero_indexes = []
                            delete_index = set()
                            temp_index = list(range(N))
                            for i in range(total_tasks):
                                temp_Y = whole_data[:, n - i]
                                for j in range(len(temp_Y)):
                                    if temp_Y[j] == 0:
                                        delete_index.add(j)
                                # temp_Y = np.delete(temp_Y,[delete_index],axis=0)
                            non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))

                            for i in range(total_tasks):
                                weights = mutual_info_regression(whole_data[non_zero_indexes, 0:N_features],
                                                                 whole_data[non_zero_indexes, n - i], random_state=0)
                                data_all.append(weights)

                            # process testing data
                            testing_index = sample(list(non_zero_indexes), N_test)
                            non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)
                            # print(non_zero_indexes)

                            training_index = sample(list(non_zero_indexes), N_train)
                            # print('testing data: ', testing_index)
                            print('training data: ', training_index)

                            tasks = meta_tasks

                            print('Main task: {}, meta tasks: {}'.format(main_task, tasks))
                            N_task = len(tasks)  # the number of tasks to group

                            # for each meta-task
                            # for i_task, task in enumerate(tasks):

                            X_train = []
                            Y_train = []
                            X_test = []
                            Y_test = []
                            X_train1 = []
                            X_train2 = []
                            Y_train1 = []
                            Y_train2 = []
                            max_X = []
                            max_Y = []
                            max_Y_all = []
                            layers = []
                            N_trains = []
                            N_tests = []
                            layers = []
                            config = []
                            lr_opt = []
                            lambdas = []
                            weights = []
                            bias = []
                            deepperf_model = None

                            ### process the training x
                            temp_X = whole_data[training_index, 0:N_features]
                            # scale x
                            temp_max_X = np.amax(temp_X, axis=0)
                            if 0 in temp_max_X:
                                temp_max_X[temp_max_X == 0] = 1
                            temp_X = np.divide(temp_X, temp_max_X)
                            X_train = np.array(temp_X)

                            # Split train data into 2 parts (67-33)
                            N_cross = int(np.ceil(len(temp_X) * 2 / 3))
                            X_train1 = (temp_X[0:N_cross, :])
                            X_train2 = (temp_X[N_cross:len(temp_X), :])

                            ### process y
                            ### process the training x
                            temp_Y = whole_data[training_index, n - main_task][:, np.newaxis]
                            # scale y
                            temp_max_Y = np.max(temp_Y) / 100
                            if temp_max_Y == 0:
                                temp_max_Y = 1
                            temp_Y = np.divide(temp_Y, temp_max_Y)
                            Y_train = np.array(temp_Y)

                            # Split train data into 2 parts (67-33)
                            Y_train1 = (temp_Y[0:N_cross, :])
                            Y_train2 = (temp_Y[N_cross:len(temp_Y), :])

                            Y_train = np.array(Y_train)
                            # Y_test = np.array(Y_test)
                            Y_test = whole_data[testing_index, n - main_task][:, np.newaxis]

                            ### process the testing x
                            X_test = np.divide(whole_data[testing_index, 0:N_features], temp_max_X)
                            X_test = np.array(X_test)

                            print(
                                '...Training the main task {} with meta model {} with {} meta samples...'.format(
                                    main_task,
                                    tasks,
                                    meta_samples))

                            reading_file_weights = 'Models/Meta_weights_{}_task{}_{}.npy'.format(
                                dir_data.split('/')[1].split('.')[0],
                                meta_tasks, meta_samples)
                            reading_file_bias = 'Models/Meta_bias_{}_task{}_{}.npy'.format(
                                dir_data.split('/')[1].split('.')[0],
                                meta_tasks, meta_samples)

                            if (os.path.exists(reading_file_weights) and os.path.exists(reading_file_bias)):
                                print('Reading meta model from: {}'.format(reading_file_weights))
                                weights = np.load(reading_file_weights, allow_pickle=True)
                                bias = np.load(reading_file_bias, allow_pickle=True)
                                # print('Rearding weights and bias of meta task {}...'.format(task))
                                print('N_layer: {}'.format(weights.shape[0]))

                                if test_mode:
                                    # for testing
                                    temp_lr_opt = 0.01
                                    n_layer_opt = weights.shape[0] - 1
                                    lambda_f = 0.123
                                    temp_config = dict()
                                    temp_config['num_neuron'] = 128
                                    temp_config['num_input'] = n - total_tasks + 1
                                    temp_config['num_layer'] = n_layer_opt
                                    temp_config['lambda'] = lambda_f
                                    temp_config['verbose'] = 1
                                    config.append(temp_config)
                                    lr_opt.append(temp_lr_opt)
                                    layers.append(n_layer_opt)

                                    # print('Training...')
                                    # # train deepperf model
                                    deepperf_model = MTLSparseModel(temp_config)
                                    if read_meta_model:
                                        deepperf_model.read_weights(weights, bias)
                                    deepperf_model.build_train()

                                    deepperf_model.train(X_train, Y_train, temp_lr_opt, max_epoch=2000)

                                    # weights, bias = deepperf_model.get_weights()
                                else:
                                    print('Step 1: Tuning the number of layers and the learning rate ...')
                                    temp_config = dict()
                                    temp_config['num_input'] = N_features
                                    temp_config['num_neuron'] = 128
                                    temp_config['lambda'] = 'NA'
                                    temp_config['decay'] = 'NA'
                                    temp_config['verbose'] = 0
                                    dir_output = 'C:/Users/Downloads'
                                    abs_error_all = np.zeros((20, 4))
                                    abs_error_all_train = np.zeros((20, 4))
                                    abs_error_layer_lr = np.zeros((20, 2))
                                    abs_err_layer_lr_min = 100
                                    count = 0
                                    layer_range = [weights.shape[0] - 1]
                                    lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
                                    for n_layer in layer_range:
                                        temp_config['num_layer'] = n_layer
                                        for lr_index, lr_initial in enumerate(lr_range):
                                            model = MTLPlainModel(temp_config)
                                            if read_meta_model:
                                                # print((weights[0].shape))
                                                model.read_weights(weights, bias)
                                            model.build_train()
                                            model.train(X_train1, Y_train1, lr_initial, max_epoch=2000)

                                            Y_pred_train = model.predict(X_train1)
                                            abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                                            abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                            Y_pred_val = model.predict(X_train2)
                                            abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                                            abs_error_all[int(n_layer), lr_index] = abs_error

                                        # Pick the learning rate that has the smallest train cost
                                        # Save testing abs_error correspond to the chosen learning_rate
                                        temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                                        temp_idx = np.where(abs(temp) < 0.0001)[0]
                                        if len(temp_idx) > 0:
                                            lr_best = lr_range[np.max(temp_idx)]
                                            err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
                                        else:
                                            lr_best = lr_range[np.argmin(temp)]
                                            err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

                                        abs_error_layer_lr[int(n_layer), 0] = err_val_best
                                        abs_error_layer_lr[int(n_layer), 1] = lr_best

                                        if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                                            abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                                                 np.argmin(temp)]
                                            count = 0
                                        else:
                                            count += 1

                                        if count >= 2:
                                            break
                                    abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

                                    # Get the optimal number of layers
                                    n_layer_opt = weights.shape[0] - 1

                                    # Find the optimal learning rate of the specific layer
                                    temp_config['num_layer'] = n_layer_opt
                                    for lr_index, lr_initial in enumerate(lr_range):
                                        model = MTLPlainModel(temp_config)
                                        if read_meta_model:
                                            model.read_weights(weights, bias)
                                        model.build_train()
                                        model.train(X_train1, Y_train1, lr_initial, max_epoch=2000)

                                        Y_pred_train = model.predict(X_train1)
                                        abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                                        abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                        Y_pred_val = model.predict(X_train2)
                                        abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                                        abs_error_all[int(n_layer), lr_index] = abs_error

                                    temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                                    temp_idx = np.where(abs(temp) < 0.0001)[0]
                                    if len(temp_idx) > 0:
                                        lr_best = lr_range[np.max(temp_idx)]
                                    else:
                                        lr_best = lr_range[np.argmin(temp)]

                                    temp_lr_opt = lr_best
                                    # print('The optimal number of layers: {}'.format(n_layer_opt))
                                    print('The optimal learning rate: {:.4f}'.format(temp_lr_opt))

                                    print('Step 2: Tuning the l1 regularized hyperparameter ...')
                                    # Use grid search to find the right value of lambda
                                    lambda_range = np.logspace(-2, np.log10(100), 30)
                                    error_min = np.zeros((1, len(lambda_range)))
                                    rel_error_min = np.zeros((1, len(lambda_range)))
                                    decay = 'NA'
                                    for idx, lambd in enumerate(lambda_range):
                                        temp_config = dict()
                                        temp_config['num_input'] = X_train1.shape[1]
                                        temp_config['num_layer'] = n_layer_opt
                                        temp_config['num_neuron'] = 128
                                        temp_config['lambda'] = lambd
                                        temp_config['verbose'] = 0
                                        model = MTLSparseModel(temp_config)
                                        if read_meta_model:
                                            # print(lambd)
                                            # print(len(weights[1]))
                                            model.read_weights(weights, bias)
                                        model.build_train()
                                        model.train(X_train1, Y_train1, temp_lr_opt, max_epoch=2000)

                                        Y_pred_val = model.predict(X_train2)
                                        val_abserror = np.mean(np.abs(Y_pred_val - Y_train2))
                                        val_relerror = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val,
                                                                                Y_train2)))

                                        error_min[0, idx] = val_abserror
                                        rel_error_min[0, idx] = val_relerror

                                    # Find the value of lambda that minimize error_min
                                    lambda_f = lambda_range[np.argmin(error_min)]
                                    print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

                                    temp_config = dict()
                                    temp_config['num_neuron'] = 128
                                    temp_config['num_input'] = n - total_tasks + 1
                                    temp_config['num_layer'] = n_layer_opt
                                    temp_config['lambda'] = lambda_f
                                    temp_config['verbose'] = 1

                                    # config.append(temp_config)
                                    # lr_opt.append(temp_lr_opt)
                                    # layers.append(n_layer_opt)
                                    # lambdas.append(lambda_f)

                                    print('Training...')
                                    # train deepperf model
                                    deepperf_model = MTLSparseModel(temp_config)
                                    if read_meta_model:
                                        deepperf_model.read_weights(weights, bias)
                                    deepperf_model.build_train()

                                    deepperf_model.train(X_train, Y_train, temp_lr_opt, max_epoch=2000)
                                    # weights, bias = deepperf_model.get_weights()

                                end_time = time.time()
                                training_time = ((end_time - start_time) / 60)
                                # test result
                                rel_error = []
                                print('\nTesting...')
                                # print(X_test)
                                Y_pred_test = deepperf_model.predict(X_test)
                                # print(Y_pred_test)
                                Y1_pred_test = temp_max_Y * Y_pred_test
                                rel_error = np.mean(
                                    np.abs(np.divide(Y_test.ravel() - Y1_pred_test.ravel(), Y_test.ravel()))) * 100
                                # results_deepperf.append(rel_error)
                                print('> Task {} Meta-learning MRE: {}'.format(main_task, rel_error))
                                print('Training time (min): {}\n'.format(training_time))

                            else:
                                print('Meta model not found')

                            if save_file:
                                with open(saving_file_name, 'a') as f:  # save the results
                                    f.write('\nTask {} MTL RE: {}'.format(main_task, rel_error))
                                    f.write('\ntime (min): {}'.format(training_time))
