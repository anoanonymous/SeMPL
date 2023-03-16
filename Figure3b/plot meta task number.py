from scipy.stats import chisquare
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import itertools as it
from bisect import bisect_left
from typing import List
import scipy.stats as ss
from pandas import Categorical
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import matplotlib.colors as colors
import matplotlib.cm as cm


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude

def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """
    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.
    :return: stats : pandas DataFrame of effect sizes
    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude
    """

    x = data.copy()
    if sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array([VD_A(list(x[val_col][x[group_col] == groups[i]].values),
                        list(x[val_col][x[group_col] == groups[j]].values)) for i, j in zip(g1, g2)])

    return pd.DataFrame({
        'A': np.unique(data[group_col])[g1],
        'B': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })

if __name__ == '__main__':

    mtl_file_names = ['Meta_x264-diff_input-10tasks_main9_meta[8]_2_24-201_01-08_02-46-58.txt'
                     ,'Meta_x264-diff_input-10tasks_main9_meta[0, 8]_2_24-201_01-08_03-28-37.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[4, 0, 8]_2_24-201_01-08_03-16-03.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[2, 4, 0, 8]_2_24-201_01-08_03-11-54.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[6, 2, 4, 0, 8]_2_24-201_01-08_03-26-25.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[7, 6, 2, 4, 0, 8]_2_24-201_01-08_04-41-01.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[1, 7, 6, 2, 4, 0, 8]_2_24-201_01-08_04-41-44.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[3, 1, 7, 6, 2, 4, 0, 8]_2_24-201_01-08_04-46-54.txt'
                    , 'Meta_x264-diff_input-10tasks_main9_meta[5, 3, 1, 7, 6, 2, 4, 0, 8]_2_24-201_01-03_20-24-49.txt'
                      ]

    main_tasks = [9]
    # main_tasks = list(range(10))

    if main_tasks==[]:
        for i_name, mtl_file_name in enumerate(mtl_file_names):
            if 'main' in mtl_file_name:
                main_task = mtl_file_name.split('main')[1].split('_')[0]
                main_tasks = main_task


    for main_task in main_tasks:
        print('\nMain task: ', main_task)
        data = {}
        for i_name, mtl_file_name in enumerate(mtl_file_names):
            raw_data = pd.read_csv(mtl_file_name)
            learning_model = mtl_file_name.split('_')[0]

            if 'main' in mtl_file_name:
                meta_model = mtl_file_name.split('main')[1].split('_')[1].replace('meta', '')
            else:
                meta_model = ''
            # print(meta_model)
            for i in range(raw_data.shape[0]):
                if len(''.join(raw_data.values[i]).split(' ')) > 3:
                    if int(''.join(raw_data.values[i]).split(' ')[1]) == main_task:

                        value = float(''.join(raw_data.values[i]).split(':')[1].strip())
                        if '{}-{}'.format(learning_model, meta_model) not in data:
                            data['{}-{}'.format(learning_model, meta_model)] = [value]
                        else:
                            data['{}-{}'.format(learning_model, meta_model)].append(value)

        # from rpy2.robjects.packages import importr
        # from rpy2.robjects import r, pandas2ri
        #
        # data = pd.DataFrame(data)
        # pandas2ri.activate()
        # sk = importr('ScottKnottESD')
        # print(data)
        # r_sk = sk.sk_esd(data)  # get the rankings
        # print(r_sk)
        # task_sk = np.array(r_sk)[3]
        # groups_sk = np.array(r_sk)[1]

        for meta_model in data:
            Q1 = np.percentile(data[meta_model], 25, interpolation='midpoint')
            Q3 = np.percentile(data[meta_model], 75, interpolation='midpoint')
            IQR = Q3 - Q1
            print('{}: {:.2f}({:.2f})'.format(meta_model, np.median(data[meta_model]), IQR))


        plt.figure(figsize=(2, 2))
        x = []
        y = []
        mres = []

        for meta_model in data:
            # print(len(meta_model.split(',')))


            x.append((len(meta_model.split(','))))
            # y.append(np.median(mre[key]))
            y.append(np.median(data[meta_model]))
            mres.append(data[meta_model])

        colormap = cm.Blues
        colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.95, len(x))]
        markers = ['o-', 'v-', 's-', 'p-', 'D-', '^-', '.-', '<-', 'd-', '>-', 'P-', 'h-', '*-', 'H-']
        plt.plot(x, y, color=colorlist[3], linewidth=1)

        Q1s = []
        Q3s = []
        for i in range(len(y)):
            # print(len(box_plot_mre[i]))
            Q1 = np.percentile(mres[i], 25, interpolation='midpoint')
            Q3 = np.percentile(mres[i], 75, interpolation='midpoint')
            Q1s.append(Q1)
            Q3s.append(Q3)
            # IQR = Q3 - Q1
            start = (x[i], x[i])
            end = (Q1, Q3)
            # median = np.median(box_plot_mre[i])
            # print(median)
            # print(Q1,Q3,IQR)
            # plt.plot(start, end)
        # print(Q1s)
        plt.fill_between(x, Q1s, Q3s,
                         color=colorlist[2], alpha=0.2)

        # plt.legend(loc='best', fontsize=7)
        # # plt.annotate('depth1_Deepperf_sub',(x, y))
        plt.xlim((int(np.min(x)), int(np.max(x))))
        plt.xticks(x)

        # colormap = cm.gist_ncar
        # colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(mre.keys()))]
        # markers = ['o-', 'v-', 's-', 'p-', 'D-', '^-', '.-', '<-', 'd-', '>-', 'P-', 'h-', '*-', 'H-']
        # # plt.plot(x, y, markers[i], color=colorlist[i], label=sys)
        # plt.boxplot(y, labels=x, vert=True, showmeans=True)
        # # plt.plot(x, y, 'o-',color=colorlist[i], label=sys)
        # # plt.legend(loc='best', fontsize=7)
        # # plt.annotate('depth1_Deepperf_sub',(x, y))
        plt.xlabel("Depth")
        plt.ylabel("MRE")
        # plt.savefig('./{}_depths.pdf'.format(sys), dpi=300, bbox_inches='tight')
        plt.show()


    #
    # sys_names = []
    #
    # mre = {}
    # time = {}
    # depth_rank = {}
    # depth_mre = {}
    #
    # for file in files:
    #     sys_name=file.split('_')[3]
    #     sys_names.append(sys_name)
    #     print('Reading {} data...'.format(sys_name))
    #     raw_data = pd.read_csv(file)
    #     # print(raw_data.shape)
    #     # print(raw_data.values)
    #     #
    #     # mre = {}
    #     # time = {}
    #     data = []
    #     baseline_mre = {}
    #     baseline_time = {}
    #
    #     # for regression_mod in ['Deepperf', 'KNN', 'SVR', 'DT', 'LR', 'KR', 'RF']:
    #     for regression_mod in ['Deepperf']:
    #         temp_mre = {}
    #         temp_time = {}
    #         for i in range(raw_data.shape[0]):
    #             if len(''.join(raw_data.values[i]).split(' '))>1:
    #                 if ''.join(raw_data.values[i]).split(' ')[1] == '{}-DaL'.format(regression_mod):
    #
    #                     if '{}_{}_{}_sub'.format(sys_name,''.join(raw_data.values[i]).split(' ')[0], regression_mod) not in temp_mre:
    #                         temp_mre['{}_{}_{}_sub'.format(sys_name,''.join(raw_data.values[i]).split(' ')[0], regression_mod)] = [float(''.join(raw_data.values[i]).split(':')[1].strip())]
    #                     else:
    #                         temp_mre['{}_{}_{}_sub'.format(sys_name,''.join(raw_data.values[i]).split(' ')[0], regression_mod)].append(float(''.join(raw_data.values[i]).split(':')[1].strip()))
    #                     # temp_mtl.append(float(''.join(raw_data.values[i]).split(':')[1].strip()))
    #
    #                 elif ''.join(raw_data.values[i]).split(' ')[1] == '{}-DaL_time'.format(regression_mod):
    #                     if '{}_{}_{}_sub'.format(sys_name,''.join(raw_data.values[i]).split(' ')[0], regression_mod) not in temp_time:
    #                         temp_time['{}_{}_{}_sub'.format(sys_name,''.join(raw_data.values[i]).split(' ')[0], regression_mod)] = [
    #                             float(''.join(raw_data.values[i]).split(':')[1].strip())]
    #                     else:
    #                         temp_time['{}_{}_{}_sub'.format(sys_name,''.join(raw_data.values[i]).split(' ')[0], regression_mod)].append(
    #                             float(''.join(raw_data.values[i]).split(':')[1].strip()))
    #
    #                 # temp_time['{}_{}_sub'.format(''.join(raw_data.values[i]).split(' ')[0], regression_mod)] = float(''.join(raw_data.values[i]).split(':')[1].strip())
    #         # print(temp_mre)
    #         mre.update(temp_mre)
    #         time.update(temp_time)
    #         mre.update(baseline_mre)
    #         time.update(baseline_time)
    #
    #         # print(mre)
    #     for key in mre.keys():
    #         Q1 = np.percentile(mre[key], 25, interpolation='midpoint')
    #         Q3 = np.percentile(mre[key], 75, interpolation='midpoint')
    #         IQR = Q3 - Q1
    #         print('{} Median(IQR) ({} runs): {:.2f}({:.2f}) ---'.format(key, len(mre[key]), np.median(mre[key]), IQR))
    #         # print('{} MRE ({} runs) : {:.2f}'.format(key, len(temp_mre[key]), np.mean(temp_mre[key])))
    #         # print('{} mean time: {:.2f}'.format(key, np.mean(temp_time[key])))
    #         # if key == 'depth1_Deepperf_sub':
    #         #     print('Wilcoxon p: {:.2f}'.format(stats.wilcoxon(baseline_mre['Deepperf'], temp_mre[key])[1]))
    #         #     print('Ranksums p: {:.2f}'.format(stats.ranksums(baseline_mre['Deepperf'], temp_mre[key])[1]))
    #         #     print('Effect size: {:.2f},{}'.format(VD_A(baseline_mre['Deepperf'], temp_mre[key])[0], VD_A(baseline_mre['Deepperf'], temp_mre[key])[1]))
    #
    #
    #     data = pd.DataFrame(temp_mre)
    #
    #     # print(data)
    #
    # # for i, sys in enumerate(sys_names):
    #     plt.figure(figsize=(2, 2))
    #     x = []
    #     y = []
    #     mres = []
    #
    #     x.append(np.median(int(key.split('_')[1].split('depth')[1])))
    #     # y.append(np.median(mre[key]))
    #     y.append(np.median(mre[key]))
    #     mres.append(mre[key])
    #
    #     colormap = cm.Blues
    #     colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.95, len(x))]
    #     markers = ['o-', 'v-', 's-', 'p-', 'D-', '^-', '.-', '<-', 'd-', '>-', 'P-', 'h-', '*-', 'H-']
    #     plt.plot(x, y, color=colorlist[3], linewidth=1)
    #
    #     Q1s = []
    #     Q3s = []
    #     for i in range(len(y)):
    #         # print(len(box_plot_mre[i]))
    #         Q1 = np.percentile(mres[i], 25, interpolation='midpoint')
    #         Q3 = np.percentile(mres[i], 75, interpolation='midpoint')
    #         Q1s.append(Q1)
    #         Q3s.append(Q3)
    #         # IQR = Q3 - Q1
    #         start = (x[i], x[i])
    #         end = (Q1, Q3)
    #         # median = np.median(box_plot_mre[i])
    #         # print(median)
    #         # print(Q1,Q3,IQR)
    #         # plt.plot(start, end)
    #     # print(Q1s)
    #     plt.fill_between(x, Q1s, Q3s,
    #                          color=colorlist[2], alpha=0.2)
    #
    #     # plt.legend(loc='best', fontsize=7)
    #     # # plt.annotate('depth1_Deepperf_sub',(x, y))
    #     plt.xlim((int(np.min(x)), int(np.max(x))))
    #     plt.xticks([1,2,3,4])
    #
    #     # colormap = cm.gist_ncar
    #     # colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(mre.keys()))]
    #     # markers = ['o-', 'v-', 's-', 'p-', 'D-', '^-', '.-', '<-', 'd-', '>-', 'P-', 'h-', '*-', 'H-']
    #     # # plt.plot(x, y, markers[i], color=colorlist[i], label=sys)
    #     # plt.boxplot(y, labels=x, vert=True, showmeans=True)
    #     # # plt.plot(x, y, 'o-',color=colorlist[i], label=sys)
    #     # # plt.legend(loc='best', fontsize=7)
    #     # # plt.annotate('depth1_Deepperf_sub',(x, y))
    #     plt.xlabel("Depth")
    #     plt.ylabel("MRE")
    #     plt.savefig('./{}_depths.pdf'.format(sys), dpi=300, bbox_inches='tight')
    #     plt.show()
    #







