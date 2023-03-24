# -*- coding: utf-8 -*-
# group according to scott knott test of training samples, instead of all samples
import numpy as np
from numpy import genfromtxt
import time
import random
from random import sample
import copy
import pandas as pd
from collections import Counter

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


def nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer, lambd, lr_initial):
    """
    Args:
        X_train1: train input data (2/3 of the whole training data)
        Y_train1: train output data (2/3 of the whole training data)
        X_train2: validate input data (1/3 of the whole training data)
        Y_train2: validate output data (1/3 of the whole training data)
        n_layer: number of layers of the neural network
        lambd: regularized parameter

    """
    config = dict()
    config['num_input'] = X_train1.shape[1]
    config['num_layer'] = n_layer
    config['num_neuron'] = 128
    config['lambda'] = lambd
    config['verbose'] = 0

    dir_output = 'C:/Users/Downloads/'

    # Build and train model
    model = MLPSparseModel(config)
    model.build_train()
    # print(X_train1[0:3])
    # print(Y_train1[0:3])
    # print(X_train2[0:3])
    # print(Y_train2[0:3])
    model.train(X_train1, Y_train1, lr_initial)

    # Evaluate trained model on validation data
    Y_pred_val = model.predict(X_train2)
    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
    rel_error = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val, Y_train2)))

    return abs_error, rel_error


# def LR_test():


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
                 'Data/sac_srad_others-5tasks.csv', 'Data/gcc-diff_input-5tasks.csv', 'Data/spear-10286-3tasks.csv',
                 'Data/storm-obj2-3tasks.csv', 'Data/imagemagick-4tasks.csv', 'Data/xz-5tasks.csv',
                 'Data/x264-diff_input-10tasks.csv']
    non_selected_dir = [0, 1, 2, 3, 5, 6, 7, 8]
    for dir, dir_data in enumerate(dir_datas):
        if dir not in non_selected_dir:
            # for dir_data in ['Data/sqlite-overwritebatch-4tasks.csv']:
            total_tasks = int(dir_data.split('tasks')[0].split('-')[-1])
            print('\nDataset: ' + dir_data)
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
            show_plot = False
            # selected_tasks = range(0,11)
            selected_tasks = [0,1,2]
            learning_model = 'deepperf'
            optimization_model = 'LR'
            # learning_model = 'rf'
            seed = 2
            N_meta_tasks = total_tasks - 1
            # N_meta_tasks = 1
            N_train = int(1 * N)
            N_test = 0
            if N_train > len(non_zero_indexes):
                # N_test = int(len(non_zero_indexes) * 3 / 10)
                N_train = (len(non_zero_indexes))
            # elif (N_train > 2):
            #     # N_test = int(N_train * 3 / 7)
            #     N_test = (len(non_zero_indexes) - N_train)
            #     # N_test = int(N * 3 / 10)
            # else:
            #     N_test = 1
            print("Training sample size: {}".format(N_train))
            print('Testing sample size: ', N_test)
            # tasks = [3,9] # the index of the tasks to group
            N_experiments = 1
            start = 0
            print('Number of expriments: ', N_experiments)

            # saving_file_name = 'Meta-learning_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed, N_train)
            print('Total sample size: ', N)
            N_features = n + 1 - total_tasks
            print('Number of features: ', N_features)

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

            results_deepperf = []
            results_mtl = []
            time_mtl = []
            time_deepperf = []

            # Start measure time
            random.seed(seed)

            # generating training and testing data
            training_index = []
            testing_index = []
            # X_train = []
            # Y_train = []
            # X_test = []
            # Y_test = []
            # max_X = []
            # max_Y = []
            # layers = []
            # N_trains = []
            # N_tests = []
            # X_train1 = []
            # X_train2 = []
            # Y_train1 = []
            # Y_train2 = []
            data = []
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

            # process testing data
            testing_index = sample(list(non_zero_indexes), N_test)
            non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)
            # print(non_zero_indexes)

            training_index = sample(list(non_zero_indexes), N_train)
            # print('testing data: ', testing_index)
            # print('training data: ', training_index)

            saving_best_meta = 'best_meta_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed, N_train)
            print('Reading best meta models from {}'.format(saving_best_meta))

            meta_to_train = {}
            with open(saving_best_meta, 'r') as f:  # save the results
                lines = f.readlines()[0]
                import ast

                meta_to_train = ast.literal_eval(lines)
            print(meta_to_train)

            random.seed(seed)

            for main_task in meta_to_train:
                if int(main_task) in selected_tasks:
                    for tasks in meta_to_train[main_task]:
                        tasks = tasks[(total_tasks - N_meta_tasks - 1):]
                        print('...Training the meta model {} for main task {}...'.format(tasks, main_task))

                        # tasks = copy.deepcopy(tasks_all)
                        N_task = len(tasks)  # the number of tasks to group

                        saving_file_weights = 'Models/Meta_weights_{}_task{}_{}.npy'.format(
                            dir_data.split('/')[1].split('.')[0],
                            tasks, N_train)
                        saving_file_bias = 'Models/Meta_bias_{}_task{}_{}.npy'.format(
                            dir_data.split('/')[1].split('.')[0],
                            tasks, N_train)

                        import os

                        if not (os.path.exists(saving_file_weights) and os.path.exists(saving_file_bias)):
                            # new_weights = np.load(saving_file_name,allow_pickle=True)
                            # print(new_weights)
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

                            for i_task, task in enumerate(tasks):
                                temp_Y = whole_data[training_index, n - task][:, np.newaxis]
                                # Scale y
                                temp_max_Y = np.max(temp_Y) / 100
                                if temp_max_Y == 0:
                                    temp_max_Y = 1
                                max_Y_all.append(temp_max_Y)  # save the max_y to a vector
                                temp_Y = np.divide(temp_Y, temp_max_Y)
                                # Split train data into 2 parts (67-33)
                                if i_task == 0:
                                    Y_train = temp_Y
                                    Y_train1 = temp_Y[0:N_cross, :]
                                    Y_train2 = temp_Y[N_cross:len(temp_X), :]
                                else:
                                    Y_train = np.hstack((Y_train, temp_Y))
                                    Y_train1 = np.hstack((Y_train1, temp_Y[0:N_cross, :]))
                                    Y_train2 = np.hstack((Y_train2, temp_Y[N_cross:len(temp_X), :]))

                            Y_train = np.array(Y_train)
                            Y_test = whole_data[testing_index, n][:, np.newaxis]
                            # Y_test = np.array(Y_test)

                            ### process the testing x
                            X_test = np.divide(whole_data[testing_index, 0:N_features], temp_max_X)
                            X_test = np.array(X_test)
                            print('\nMeta sample size: {}'.format(len(X_train)))
                            ### training deepperf
                            if learning_model == 'deepperf':

                                if test_mode:
                                    # for testing
                                    for i in range(0, N_task):
                                        print('---Training the meta task {}---'.format(tasks[i]))
                                        temp_lr_opt = 0.123
                                        n_layer_opt = 5
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
                                        if i == 0:
                                            deepperf_model = MTLSparseModel(config[i])
                                            deepperf_model.build_train()
                                        else:
                                            deepperf_model = MTLSparseModel(config[i])
                                            deepperf_model.read_weights(weights, bias)
                                            deepperf_model.build_train()

                                        deepperf_model.train(X_train, Y_train[:, i][:, np.newaxis], lr_opt[i],
                                                             max_epoch=2000)
                                        weights, bias = deepperf_model.get_weights()
                                else:
                                    for i in range(0, N_task):
                                        print('\n---Tuning task {}---'.format(tasks[i]))
                                        if i == 0:
                                            # for testing, start to comment here
                                            print('Tuning hyperparameters...')
                                            print('Step 1: Tuning the number of layers and the learning rate ...')
                                            temp_config = dict()
                                            temp_config['num_input'] = n - total_tasks + 1
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
                                            layer_range = range(2, 15)
                                            lr_range = [0.0001, 0.001, 0.01, 0.1]
                                            # lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 5)
                                            for n_layer in layer_range:
                                                temp_config['num_layer'] = n_layer
                                                for lr_index, lr_initial in enumerate(lr_range):
                                                    model = MLPPlainModel(temp_config)
                                                    model.build_train()
                                                    model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial)

                                                    Y_pred_train = model.predict(X_train1)
                                                    abs_error_train = np.mean(
                                                        np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                                    Y_pred_val = model.predict(X_train2)
                                                    abs_error = np.mean(
                                                        np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                                    abs_error_all[int(n_layer), lr_index] = abs_error

                                                # Pick the learning rate that has the smallest train cost
                                                # Save testing abs_error correspond to the chosen learning_rate
                                                temp = abs_error_all_train[int(n_layer), :] / np.max(
                                                    abs_error_all_train)
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
                                            n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 5

                                            # Find the optimal learning rate of the specific layer
                                            temp_config['num_layer'] = n_layer_opt
                                            for lr_index, lr_initial in enumerate(lr_range):
                                                model = MLPPlainModel(temp_config)
                                                model.build_train()
                                                model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial)

                                                Y_pred_train = model.predict(X_train1)
                                                abs_error_train = np.mean(
                                                    np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                                abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                                Y_pred_val = model.predict(X_train2)
                                                abs_error = np.mean(np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                                abs_error_all[int(n_layer), lr_index] = abs_error

                                            temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                                            temp_idx = np.where(abs(temp) < 0.0001)[0]
                                            if len(temp_idx) > 0:
                                                lr_best = lr_range[np.max(temp_idx)]
                                            else:
                                                lr_best = lr_range[np.argmin(temp)]

                                            temp_lr_opt = lr_best
                                            print('The optimal number of layers: {}'.format(n_layer_opt))
                                            print('The optimal learning rate: {:.4f}'.format(temp_lr_opt))

                                            print('Step 2: Tuning the l1 regularized hyperparameter ...')
                                            # Use grid search to find the right value of lambda
                                            lambda_range = np.logspace(-2, np.log10(100), 30)
                                            error_min = np.zeros((1, len(lambda_range)))
                                            rel_error_min = np.zeros((1, len(lambda_range)))
                                            decay = 'NA'
                                            for idx, lambd in enumerate(lambda_range):
                                                # val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1[:, i][:, np.newaxis],
                                                #                                        X_train2, Y_train2[:, i][:, np.newaxis],
                                                #                                        n_layer_opt, lambd, temp_lr_opt)
                                                temp_config = dict()
                                                temp_config['num_input'] = X_train1.shape[1]
                                                temp_config['num_layer'] = n_layer_opt
                                                temp_config['num_neuron'] = 128
                                                temp_config['lambda'] = lambd
                                                temp_config['verbose'] = 0
                                                model = MLPSparseModel(temp_config)
                                                model.build_train()
                                                model.train(X_train1, Y_train1[:, i][:, np.newaxis], temp_lr_opt)

                                                Y_pred_val = model.predict(X_train2)
                                                val_abserror = np.mean(
                                                    np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                                val_relerror = np.mean(np.abs(
                                                    np.divide(Y_train2[:, i][:, np.newaxis] - Y_pred_val,
                                                              Y_train2[:, i][:, np.newaxis])))

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

                                            config.append(temp_config)
                                            lr_opt.append(temp_lr_opt)
                                            layers.append(n_layer_opt)
                                            lambdas.append(lambda_f)

                                            print('Training...')
                                            # train deepperf model
                                            deepperf_model = MTLSparseModel(config[i])
                                            deepperf_model.build_train()

                                        else:
                                            print('Step 1: Tuning the number of layers and the learning rate ...')
                                            temp_config = dict()
                                            temp_config['num_input'] = n - total_tasks + 1
                                            temp_config['num_neuron'] = 128
                                            temp_config['lambda'] = 'NA'
                                            temp_config['decay'] = 'NA'
                                            temp_config['verbose'] = 0
                                            dir_output = 'C:/Users/Downloads'
                                            abs_error_all = np.zeros((20, 10))
                                            abs_error_all_train = np.zeros((20, 10))
                                            abs_error_layer_lr = np.zeros((20, 2))
                                            abs_err_layer_lr_min = 100
                                            count = 0
                                            layer_range = [layers[0]]
                                            lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 10)
                                            for n_layer in layer_range:
                                                temp_config['num_layer'] = n_layer
                                                for lr_index, lr_initial in enumerate(lr_range):
                                                    model = MTLPlainModel(temp_config)
                                                    model.read_weights(weights, bias)
                                                    model.build_train()
                                                    model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial,
                                                                max_epoch=2000)

                                                    Y_pred_train = model.predict(X_train1)
                                                    abs_error_train = np.mean(
                                                        np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                                    Y_pred_val = model.predict(X_train2)
                                                    abs_error = np.mean(
                                                        np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                                    abs_error_all[int(n_layer), lr_index] = abs_error

                                                # Pick the learning rate that has the smallest train cost
                                                # Save testing abs_error correspond to the chosen learning_rate
                                                temp = abs_error_all_train[int(n_layer), :] / np.max(
                                                    abs_error_all_train)
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
                                            n_layer_opt = layers[0]

                                            # Find the optimal learning rate of the specific layer
                                            temp_config['num_layer'] = n_layer_opt
                                            for lr_index, lr_initial in enumerate(lr_range):
                                                model = MTLPlainModel(temp_config)
                                                model.read_weights(weights, bias)
                                                model.build_train()
                                                model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial,
                                                            max_epoch=2000)

                                                Y_pred_train = model.predict(X_train1)
                                                abs_error_train = np.mean(
                                                    np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                                abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                                Y_pred_val = model.predict(X_train2)
                                                abs_error = np.mean(np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
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
                                                # val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1[:, i][:, np.newaxis],
                                                #                                        X_train2, Y_train2[:, i][:, np.newaxis],
                                                #                                        n_layer_opt, lambd, temp_lr_opt)

                                                temp_config = dict()
                                                temp_config['num_input'] = X_train1.shape[1]
                                                temp_config['num_layer'] = n_layer_opt
                                                temp_config['num_neuron'] = 128
                                                temp_config['lambda'] = lambd
                                                temp_config['verbose'] = 0
                                                model = MTLSparseModel(temp_config)
                                                model.read_weights(weights, bias)
                                                model.build_train()
                                                model.train(X_train1, Y_train1[:, i][:, np.newaxis], temp_lr_opt,
                                                            max_epoch=2000)

                                                Y_pred_val = model.predict(X_train2)
                                                val_abserror = np.mean(
                                                    np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                                val_relerror = np.mean(
                                                    np.abs(np.divide(Y_train2[:, i][:, np.newaxis] - Y_pred_val,
                                                                     Y_train2[:, i][:, np.newaxis])))

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

                                            config.append(temp_config)
                                            lr_opt.append(temp_lr_opt)
                                            layers.append(n_layer_opt)
                                            lambdas.append(lambda_f)

                                            print('Training...')
                                            # train deepperf model
                                            deepperf_model = MTLSparseModel(config[i])
                                            deepperf_model.read_weights(weights, bias)
                                            deepperf_model.build_train()

                                            # temp_lr_opt = lr_opt[0]
                                            n_layer_opt = layers[0]
                                            lambda_f = lambdas[0]
                                            temp_config = config[0]
                                            config.append(temp_config)
                                            lr_opt.append(temp_lr_opt)  # new
                                            layers.append(n_layer_opt)
                                            lambdas.append(lambda_f)

                                        deepperf_model.train(X_train, Y_train[:, i][:, np.newaxis], lr_opt[i],
                                                             max_epoch=2000)
                                        weights, bias = deepperf_model.get_weights()

                                if save_file:
                                    np.save(saving_file_weights, (weights))
                                    np.save(saving_file_bias, (bias))


                        else:
                            print('{} existed'.format(saving_file_weights))