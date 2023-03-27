import numpy as np
from numpy import genfromtxt
import time
import random
from random import sample
import copy
import pandas as pd
from collections import Counter
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from Meta_sparse_model_tf2 import MTLSparseModel
from mlp_sparse_model_tf2 import MLPSparseModel
from mlp_plain_model_tf2 import MLPPlainModel


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


# Main function
def sequence_selection(dir_data):
    total_tasks = int(dir_data.split('tasks')[0].split('-')[-1])
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    (N, n) = whole_data.shape
    n = n - 1

    # delete the zero-performance samples
    delete_index = set()
    temp_index = list(range(N))
    for i in range(total_tasks):
        temp_Y = whole_data[:, n - i]
        for j in range(len(temp_Y)):
            if temp_Y[j] == 0:
                delete_index.add(j)
    non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))

    save_file = False
    selected_tasks = list(range(0,total_tasks))
    learned_tasks = []
    seed = 2
    N_train = int(1 * len(non_zero_indexes))
    LR_max_iter = 1000 # default 100

    N_experiments = 30
    start = 0
    # print('Number of expriments: ', N_experiments)

    # saving_file_name = 'Meta-learning_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed, N_train)
    # print('Total sample size: ', N)
    N_features = n + 1 - total_tasks

    # Start measure time
    random.seed(seed)

    # delete the zero-performance samples
    delete_index = set()
    temp_index = list(range(N))
    for i in range(total_tasks):
        temp_Y = whole_data[:, n - i]
        for j in range(len(temp_Y)):
            if temp_Y[j] == 0:
                delete_index.add(j)
    non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))


    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
    for task in range(total_tasks):
        whole_data[:, n-task] = min_max_scaler.fit_transform(np.array(whole_data[:, n-task]).reshape(-1, 1))[:, 0]
        # print(temp_normalised_data)


    task_sk = np.arange(0, total_tasks)
    groups_sk = np.zeros(total_tasks)
    # groups_sk = np.arange(0, total_tasks)

    counter_result = Counter(groups_sk)
    # print(task_sk)
    # print(groups_sk)
    for group in counter_result.keys():
        temp_group = []
        for i in range(len(groups_sk)):
            if groups_sk[i] == group:
                temp_group.append(task_sk[i])
        # print('Group {} tasks: {}'.format(group, temp_group))

    for group in counter_result.keys():
        # testing_index = []
        tasks_all = []
        meta_to_train = {}
        # print('Group {}:'.format(int(group)))

        for i in range(len(groups_sk)):
            if groups_sk[i] == group and task_sk[i] in selected_tasks:
                tasks_all.append(task_sk[i])
        # print(tasks)
        # tasks_all = tasks_all[::-1]

        import itertools
        import warnings
        warnings.filterwarnings("ignore")
        for main_task in selected_tasks:
            if main_task not in learned_tasks:
                groups_all = []
                tasks = copy.deepcopy(tasks_all)
                N_task = len(tasks)  # the number of tasks to group
                tasks.remove(main_task)
                # print('Target task: {}, meta tasks: {}'.format(main_task, tasks))
                # for i in range(1, len(tasks) + 1):
                for i in range(1, 2):
                    iter = itertools.permutations(tasks, i)
                    combinations = list(iter)
                    groups_all += combinations
                    # print('Combinations: {}'.format(combinations))
                # print('{} combinations in total'.format(len(groups_all)))

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

                temp_N_test = int(len(non_zero_indexes) * 3 / 10)
                temp_N_train = len(non_zero_indexes) - temp_N_test

                LR_results = {}
                best_hyperparameters = {}

                start_time = time.time()

                for temp_main_task in tasks:
                # for ne in range(start, start + N_experiments):
                    # print('\nExperiment {}: '.format(ne + 1))

                    if 'main{}'.format(temp_main_task) in LR_results:
                        temp_LR_result = LR_results['main{}'.format(temp_main_task)]
                    else:
                        temp_LR_result = {}
                    for combination in groups_all:
                        if temp_main_task not in combination:
                            # print('Using Meta{} for task{}'.format(combination, temp_main_task))
                            model = SGDRegressor(max_iter=LR_max_iter, warm_start=True, random_state=seed, penalty='l1')

                            # for temp_main_task in tasks:
                            for ne in range(start, start + N_experiments):
                                random.seed(ne * seed)

                                for temp_train_task in combination:
                                    # print('Training meta{}'.format(temp_train_task))
                                    X_train = whole_data[non_zero_indexes, 0:N_features]
                                    Y_train = whole_data[non_zero_indexes, n - temp_train_task][:, np.newaxis]

                                    # if not test_mode:
                                    #     # print('Tuning hyperparameters...')
                                    #     param = {'max_iter': [50, 100, 200],
                                    #              'learning_rate': ('adaptive', 'invscaling', 'optimal')
                                    #              }
                                    #
                                    #     gridS = GridSearchCV(model, param)
                                    #     gridS.fit(X_train, Y_train)
                                    #     model = SGDRegressor(**gridS.best_params_, warm_start=True, random_state=ne*seed,
                                    #                          penalty='l1')

                                    model.fit(X_train, Y_train)

                                # process testing data
                                temp_testing_index = sample(list(non_zero_indexes), temp_N_test)
                                temp_training_index = np.setdiff1d(non_zero_indexes, temp_testing_index)

                                X_train = whole_data[temp_training_index, 0:N_features]
                                Y_train = whole_data[temp_training_index, n - temp_main_task][:, np.newaxis]
                                X_test = whole_data[temp_testing_index, 0:N_features]
                                Y_test = whole_data[temp_testing_index, n - temp_main_task][:, np.newaxis]

                                model.fit(X_train, Y_train)

                                Y_pred_test = model.predict(X_test)

                                rel_error = np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel()))) * 100
                                # print('MRE: {}'.format(rel_error))

                                if 'meta{}'.format(combination) in temp_LR_result:
                                    temp_LR_result['meta{}'.format(combination)].append(rel_error)
                                else:
                                    temp_LR_result['meta{}'.format(combination)] = [rel_error]

                            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
                            temp_LR_result['meta{}'.format(combination)] = min_max_scaler.fit_transform(np.array(temp_LR_result['meta{}'.format(combination)]).reshape(-1, 1)).ravel().tolist()
                            # print(temp_LR_result['meta{}'.format(combination)])

                    LR_results['main{}'.format(temp_main_task)] = temp_LR_result

                # print('LR results: {}'.format(LR_results))

                temp_meta_results = {}
                # for meta in tasks:
                for meta in groups_all:
                    meta = 'meta{}'.format(meta).replace('(', '').replace(',)', '').replace(')', '')
                    # print(meta)
                    for temp_main in LR_results:
                        # print(LR_results[temp_main])
                        for temp_meta in LR_results[temp_main]:
                            # print(temp_meta.replace(',)', '').replace('(', '').replace(')', ''), meta)
                            if temp_meta.replace('(', '').replace(',)', '').replace(')', '') == meta:
                                # print(temp_meta, temp_main)
                                temp_meta_new = (temp_meta.replace(',)', '').replace('meta(', '')).replace(')', '')
                                if temp_meta_new not in temp_meta_results:
                                    temp_meta_results[temp_meta_new] = {}
                                temp_meta_results[temp_meta_new][temp_main] = LR_results[temp_main][temp_meta]
                # print('Meta task results: {}'.format(temp_meta_results))

                meta_results = {}
                for temp_meta in temp_meta_results:
                    average_results = []
                    for temp_main in temp_meta_results[temp_meta]:
                        from sklearn import preprocessing

                        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
                        temp_normalised_result = np.array(temp_meta_results[temp_meta][temp_main])
                        temp_normalised_result = min_max_scaler.fit_transform(
                            np.array(temp_meta_results[temp_meta][temp_main]).reshape(-1, 1))
                        # print(temp_meta, temp_main, temp_normalised_result.ravel(),
                        #       np.mean(temp_normalised_result.ravel()))
                        average_results += list(temp_normalised_result.ravel())
                    meta_results[temp_meta] = average_results
                # print('Meta results: ', meta_results)

                scott_scores = {}
                if len(tasks) >= 3:
                    for temp_main_task in LR_results:
                        data = pd.DataFrame(LR_results[temp_main_task])
                        from rpy2.robjects.packages import importr
                        from rpy2.robjects import r, pandas2ri
                        pandas2ri.activate()
                        sk = importr('ScottKnottESD')
                        # print(data)
                        r_sk = sk.sk_esd(data)  # get the rankings
                        # print(r_sk)
                        task_sk = np.array(r_sk)[3]
                        groups_sk = np.array(r_sk)[1]

                        max_score = np.max(groups_sk)
                        for i in range(len(groups_sk)):
                            groups_sk[i] = max_score - groups_sk[i] + 1

                        for i, task in enumerate(task_sk):
                            temp = r_sk[2][int(task_sk[i]) - 1]
                            temp_meta = []
                            for j, temp2 in enumerate(temp.replace('meta', '').split('..')):
                                temp2 = temp2.replace('.', '')
                                if temp2 != '':
                                    temp_meta.append(int(temp2))
                            # temp_meta = 'meta{}'.format(temp_meta)
                            temp_meta = '{}'.format(temp_meta).replace('[', '').replace(']', '')

                            if temp_meta not in scott_scores:
                                scott_scores[temp_meta] = [groups_sk[i]]
                            else:
                                scott_scores[temp_meta].append(groups_sk[i])
                else:
                    for temp_meta in meta_results:
                        scott_scores[temp_meta] = np.mean(meta_results[temp_meta])

                # print('scott_scores: ', scott_scores)
                best_score = 999999
                temp_best_sequence = []
                for temp_meta in scott_scores:
                    scott_scores[temp_meta] = np.mean(scott_scores[temp_meta])
                    temp_best_sequence.append(temp_meta.replace('meta', ''))

                # print('Average rankings: {}'.format(scott_scores))

                # print('The best meta model to pick for {} is meta{}'.format(main_task, temp_best_sequence))

                for temp_meta in meta_results:
                    average_results = meta_results[temp_meta]
                    Q1 = np.percentile(average_results, 25, interpolation='midpoint')
                    Q3 = np.percentile(average_results, 75, interpolation='midpoint')
                    IQR = Q3 - Q1
                    temp_meta_results[temp_meta] = [scott_scores[temp_meta], np.mean(average_results), IQR]

                temp_meta_results = sorted(temp_meta_results.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)

                # print('Average MRE: {}'.format(temp_meta_results))


                best_sequence = []
                for temp_meta in enumerate(temp_meta_results):
                    # print(temp_meta)
                    best_sequence.append(int(temp_meta[1][0].replace('meta[','').replace(']','')))
                # print(temp_meta_results.keys())

                # print('Final best meta: {}'.format(best_sequence))

                meta_to_train[main_task] = [best_sequence]

        saving_best_sequence = 'best_sequence_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed, N_train)
        end_time = time.time()

        total_time = (end_time - start_time) / 60
        # print('Total time cost: {}'.format(total_time))

        # for i_main, main_task in enumerate(meta_to_train):
        #     print('The best meta model to pick for {} is meta{}'.format(main_task, meta_to_train[main_task]))


        if save_file:
            import json
            print(json.dumps(meta_to_train))
            with open(saving_best_sequence, 'w') as f:  # save the results
                f.write(json.dumps(meta_to_train))
            print('Results saved to {}'.format(saving_best_sequence))

        return meta_to_train
