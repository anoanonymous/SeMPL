import numpy as np
from numpy import genfromtxt
import time
import random
from random import sample
import os
from utils.Meta_sparse_model_tf2 import MTLSparseModel
from sequence_selection import sequence_selection
from meta_training import meta_training
from utils.general import get_sizes, get_non_zero_indexes, process_training_data
from utils.hyperparameter_tuning import hyperparameter_tuning
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    subject_systems = ['Data/deeparch-SizeReduction-3tasks.csv','Data/sac_srad_others-5tasks.csv', 'Data/sqlite-overwritebatch-4tasks.csv',
                'Data/nginx-4tasks.csv', 'Data/spear-10286-3tasks.csv', 'Data/storm-obj2-3tasks.csv', 'Data/imagemagick-4tasks.csv',
                'Data/exastencils-4tasks.csv', 'Data/x264-diff_input-10tasks.csv']
    ########### experiment parameters ###########
    selected_sys = [6]  # set the subject systems to evaluate
    selected_sizes = [0]  # set the sizes to evaluate
    save_file = True
    test_mode = True
    read_meta_model = True
    seed = 2
    N_experiments = 3
    start = 0
    learned_tasks = []
    learned_meta_models = ['']
    ########### experiment parameters ###########
    for dir, dir_data in enumerate(subject_systems):
        if dir in selected_sys:
            system = dir_data.replace('Data/', '').replace('.csv', '')
            total_tasks = int(dir_data.split('tasks')[0].split('-')[-1])
            sample_sizes, selected_tasks = get_sizes(dir_data, total_tasks) # get the meta-tasks and fine-tuning sizes
            print('Dataset: ' + system)
            whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
            (N, n) = whole_data.shape
            n = n - 1
            N_features = n + 1 - total_tasks
            print('Number of expriments: ', N_experiments)
            print('Total sample size: {}, Number of features: {}'.format(N, N_features))
            print('Training sizes: {}, selected_tasks: {}'.format(sample_sizes, selected_tasks))
            N_meta_tasks = total_tasks - 1

            for i_size in selected_sizes:
                print('--- Evaluating {} with S_{} ---'.format(system.split('-')[0].split('_')[0], i_size+1))
                # delete the zero-performance samples
                non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)
                N_train = sample_sizes[i_size]
                N_test = (len(non_zero_indexes) - N_train)
                meta_samples = int(1 * len(non_zero_indexes))
                print('Training size: {}, testing size: {}, Meta-training size: {}'.format(N_train, N_test, meta_samples))
                if read_meta_model:
                    saving_best_meta = 'best_sequence_{}_{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], seed,meta_samples)
                    meta_to_train = {}
                    if (os.path.exists(saving_best_meta) and os.path.exists(reading_file_bias)):
                        print('> Reading best sequence from {}...'.format(saving_best_meta))
                        with open(saving_best_meta, 'r') as f:  # save the results
                            lines = f.readlines()[0]
                            import ast
                            meta_to_train = ast.literal_eval(lines)
                    else:
                        print('> Running sequence selection...')
                        meta_to_train = sequence_selection(dir_data)

                    for main_task in meta_to_train:
                        for meta_tasks in meta_to_train[main_task]:
                            meta_tasks = meta_tasks[(total_tasks - N_meta_tasks - 1):]
                            if int(main_task) not in learned_tasks and meta_tasks not in learned_meta_models:
                                saving_file_name = 'SeMPL_{}_T{}_M{}_{}-{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0],
                                    main_task, meta_tasks, N_train, meta_samples,time.strftime('%m-%d_%H-%M-%S',time.localtime(time.time())))
                                # print('saving_file_name: {}'.format(saving_file_name))
                                main_task = int(main_task)
                                if save_file:
                                    with open(saving_file_name, 'w') as f:  # save the results
                                        f.write('N_train={} N_test={}'.format(N_train, N_test))

                                print('> Meta-training {} for target task T_{}...'.format(meta_tasks, main_task+1))
                                reading_file_weights = 'Models/weights_{}_M{}_{}.npy'.format(dir_data.split('/')[1].split('.')[0],meta_tasks, meta_samples)
                                reading_file_bias = 'Models/bias_{}_M{}_{}.npy'.format(dir_data.split('/')[1].split('.')[0],meta_tasks, meta_samples)

                                if (os.path.exists(reading_file_weights) and os.path.exists(reading_file_bias)):
                                    print('\t>> Reading meta model from: {}'.format(reading_file_weights))
                                    weights = np.load(reading_file_weights, allow_pickle=True)
                                    bias = np.load(reading_file_bias, allow_pickle=True)
                                else:
                                    weights, bias = meta_training(dir_data, [main_task], meta_to_train)
                                    weights = np.array(weights)
                                    bias = np.array(bias)

                                print('> Fine-tuning...')
                                for ne in range(start, start + N_experiments):
                                    # print('\tRun {}: '.format(ne + 1))
                                    random.seed(ne * seed)
                                    N_task = len(meta_tasks)  # the number of tasks to group

                                    if save_file:
                                        with open(saving_file_name, 'a') as f:  # save the results
                                            f.write('\nRun {}'.format(ne + 1))

                                    # Start measure time
                                    start_time = time.time()

                                    # delete the zero-performance samples
                                    non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)
                                    testing_index = sample(list(non_zero_indexes), N_test)
                                    non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)
                                    training_index = sample(list(non_zero_indexes), N_train)

                                    ### process training data
                                    max_X, X_train, X_train1, X_train2, max_Y, Y_train, Y_train1, Y_train2 = process_training_data(whole_data, training_index, N_features, n, main_task)
                                    ### process testing data
                                    Y_test = whole_data[testing_index, n - main_task][:, np.newaxis]
                                    X_test = np.divide(whole_data[testing_index, 0:N_features], max_X)

                                    # default hyperparameters, just for testing
                                    if test_mode == True:
                                        lr_opt = 0.123
                                        n_layer_opt = weights.shape[0] - 1
                                        lambda_f = 0.123
                                        config = dict()
                                        config['num_neuron'] = 128
                                        config['num_input'] = N_features
                                        config['num_layer'] = n_layer_opt
                                        config['lambda'] = lambda_f
                                        config['verbose'] = 0
                                    # if not test_mode, tune the hyperparameters
                                    else:
                                        n_layer_opt = weights.shape[0] - 1
                                        lambda_f, lr_opt = hyperparameter_tuning([N_features, X_train1, Y_train1, X_train2, Y_train2, n_layer_opt])
                                        # save the hyperparameters
                                        config = dict()
                                        config['num_neuron'] = 128
                                        config['num_input'] = N_features
                                        config['num_layer'] = n_layer_opt
                                        config['lambda'] = lambda_f
                                        config['verbose'] = 0

                                    SeMPL_model = MTLSparseModel(config)
                                    if read_meta_model:
                                        SeMPL_model.read_weights(weights, bias)
                                    SeMPL_model.build_train()
                                    SeMPL_model.train(X_train, Y_train, lr_opt, max_epoch=2000)

                                    end_time = time.time()
                                    training_time = ((end_time - start_time) / 60)
                                    # test result
                                    rel_error = []
                                    Y_pred_test = SeMPL_model.predict(X_test)
                                    Y1_pred_test = max_Y * Y_pred_test
                                    rel_error = np.mean(
                                        np.abs(np.divide(Y_test.ravel() - Y1_pred_test.ravel(), Y_test.ravel()))) * 100
                                    print('\t>> Run{} {} S_{} T_{} MRE: {:.2f}, Training time (min): {:.2f}'.format(ne+1, system, i_size+1, main_task+1, rel_error, training_time))

                                    if save_file:
                                        with open(saving_file_name, 'a') as f:  # save the results
                                            f.write('\nTarget task T_{} SeMPL RE: {}'.format(main_task, rel_error))
                                            f.write('\ntime (min): {}'.format(training_time))
