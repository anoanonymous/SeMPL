import numpy as np
from numpy import genfromtxt
import time
import random
from random import sample
import copy
import pandas as pd
from collections import Counter
from sklearn.linear_model import SGDRegressor
from utils.general import get_non_zero_indexes
import itertools
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def sequence_selection(dir_data):
    total_tasks = int(dir_data.split('tasks')[0].split('-')[-1])
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    (N, n) = whole_data.shape
    n = n - 1
    # delete the zero-performance samples
    non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)
    save_file = False
    selected_tasks = list(range(0,total_tasks))
    learned_tasks = []
    seed = 2
    N_train = int(1 * len(non_zero_indexes))
    LR_max_iter = 1000 # default 100
    N_experiments = 30
    start = 0
    N_features = n + 1 - total_tasks
    random.seed(seed)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
    for task in range(total_tasks):
        whole_data[:, n-task] = min_max_scaler.fit_transform(np.array(whole_data[:, n-task]).reshape(-1, 1))[:, 0]

    task_sk = np.arange(0, total_tasks)
    groups_sk = np.zeros(total_tasks)
    counter_result = Counter(groups_sk)
    for group in counter_result.keys():
        temp_group = []
        for i in range(len(groups_sk)):
            if groups_sk[i] == group:
                temp_group.append(task_sk[i])

    for group in counter_result.keys():
        tasks_all = []
        meta_to_train = {}
        for i in range(len(groups_sk)):
            if groups_sk[i] == group and task_sk[i] in selected_tasks:
                tasks_all.append(task_sk[i])
        for main_task in selected_tasks:
            if main_task not in learned_tasks:
                groups_all = []
                tasks = copy.deepcopy(tasks_all)
                tasks.remove(main_task)
                for i in range(1, 2):
                    iter = itertools.permutations(tasks, i)
                    combinations = list(iter)
                    groups_all += combinations
                non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)
                temp_N_test = int(len(non_zero_indexes) * 3 / 10)
                LR_results = {}
                start_time = time.time()
                for temp_main_task in tasks:
                    if 'main{}'.format(temp_main_task) in LR_results:
                        temp_LR_result = LR_results['main{}'.format(temp_main_task)]
                    else:
                        temp_LR_result = {}
                    for combination in groups_all:
                        if temp_main_task not in combination:
                            model = SGDRegressor(max_iter=LR_max_iter, warm_start=True, random_state=seed, penalty='l1')
                            for ne in range(start, start + N_experiments):
                                random.seed(ne * seed)

                                for temp_train_task in combination:
                                    X_train = whole_data[non_zero_indexes, 0:N_features]
                                    Y_train = whole_data[non_zero_indexes, n - temp_train_task][:, np.newaxis]
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

                                if 'meta{}'.format(combination) in temp_LR_result:
                                    temp_LR_result['meta{}'.format(combination)].append(rel_error)
                                else:
                                    temp_LR_result['meta{}'.format(combination)] = [rel_error]

                            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
                            temp_LR_result['meta{}'.format(combination)] = min_max_scaler.fit_transform(np.array(temp_LR_result['meta{}'.format(combination)]).reshape(-1, 1)).ravel().tolist()

                    LR_results['main{}'.format(temp_main_task)] = temp_LR_result

                temp_meta_results = {}
                for meta in groups_all:
                    meta = 'meta{}'.format(meta).replace('(', '').replace(',)', '').replace(')', '')
                    for temp_main in LR_results:
                        for temp_meta in LR_results[temp_main]:
                            if temp_meta.replace('(', '').replace(',)', '').replace(')', '') == meta:
                                temp_meta_new = (temp_meta.replace(',)', '').replace('meta(', '')).replace(')', '')
                                if temp_meta_new not in temp_meta_results:
                                    temp_meta_results[temp_meta_new] = {}
                                temp_meta_results[temp_meta_new][temp_main] = LR_results[temp_main][temp_meta]
                meta_results = {}
                for temp_meta in temp_meta_results:
                    average_results = []
                    for temp_main in temp_meta_results[temp_meta]:
                        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
                        temp_normalised_result = min_max_scaler.fit_transform(
                            np.array(temp_meta_results[temp_meta][temp_main]).reshape(-1, 1))
                        average_results += list(temp_normalised_result.ravel())
                    meta_results[temp_meta] = average_results

                scott_scores = {}
                if len(tasks) >= 3:
                    for temp_main_task in LR_results:
                        data = pd.DataFrame(LR_results[temp_main_task])
                        from rpy2.robjects.packages import importr
                        from rpy2.robjects import r, pandas2ri
                        pandas2ri.activate()
                        sk = importr('ScottKnottESD')
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
                            temp_meta = '{}'.format(temp_meta).replace('[', '').replace(']', '')

                            if temp_meta not in scott_scores:
                                scott_scores[temp_meta] = [groups_sk[i]]
                            else:
                                scott_scores[temp_meta].append(groups_sk[i])
                else:
                    for temp_meta in meta_results:
                        scott_scores[temp_meta] = np.mean(meta_results[temp_meta])

                temp_best_sequence = []
                for temp_meta in scott_scores:
                    scott_scores[temp_meta] = np.mean(scott_scores[temp_meta])
                    temp_best_sequence.append(temp_meta.replace('meta', ''))

                for temp_meta in meta_results:
                    average_results = meta_results[temp_meta]
                    Q1 = np.percentile(average_results, 25, interpolation='midpoint')
                    Q3 = np.percentile(average_results, 75, interpolation='midpoint')
                    IQR = Q3 - Q1
                    temp_meta_results[temp_meta] = [scott_scores[temp_meta], np.mean(average_results), IQR]

                temp_meta_results = sorted(temp_meta_results.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
                best_sequence = []
                for temp_meta in enumerate(temp_meta_results):
                    best_sequence.append(int(temp_meta[1][0].replace('meta[','').replace(']','')))
                meta_to_train[main_task] = [best_sequence]

        saving_best_sequence = 'best_sequence_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed, N_train)
        end_time = time.time()
        total_time = (end_time - start_time) / 60
        # print('Total time cost: {}'.format(total_time))
        if save_file:
            import json
            print(json.dumps(meta_to_train))
            with open(saving_best_sequence, 'w') as f:  # save the results
                f.write(json.dumps(meta_to_train))
            print('Results saved to {}'.format(saving_best_sequence))

        return meta_to_train
