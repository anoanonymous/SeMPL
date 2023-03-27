# Predicting Software Performance in Multiple Environments with Sequential Meta-Learning
>Learning and predicting the performance of given software configurations are of high importance to many software engineering activities. While configurable software systems will almost certainly face diverse running environments (e.g., version, hardware, and workload), current work often builds performance models under a single environment, hence ignoring the rich knowledge from different settings and also restricting the accuracy for a new environment. In this paper, we target configuration performance learning under multiple environments. We do so by designing **SeMPL** —— a meta-learning framework that learns the common understanding from configurations measured in distinct environments (meta-tasks) and generalizes them to the target setting (target-task). What makes it unique is that unlike common meta-learning frameworks (e.g., MAML and MetaSGD) that trains the meta-tasks in parallel during pre-training, we design sequential meta-training that is able to discriminate the contributions of meta-tasks in the meta-model built via ordering them, which fits better with the characteristic of configuration data that is known to dramatically differ between different environments/tasks. Through comparing with 15 state-of-the-art models under 9 systems, our extensive experimental results demonstrate that SeMPL performs considerably better in **89%** of the cases with up to $99\%$ accuracy improvement, while being data-efficient, leading to at most **3.86×** speedup.
> 
This repository contains the **key codes**, **full data used**, **raw experiment results** and **the supplementary tables** for the paper.

# Documents
- **Codes**

    └─ **SeMPL_main.py**: 
the *main program* for using SeMPL, which automatically reads data from csv files, trains and evaluates, and save the results.

    └─ **mlp_plain_model.py**:
contains functions to construct and train plain DNN. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf). 

    └─ **mlp_sparse_model.py**:
contains functions to construct and build DNN with L1 regularization. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).

    └─ **general.py**:
    contains utility functions to build DNN and other ML models.
    
    └─ **hyperparameter_tuning.py**:
    contains the function that efficiently tunes hyperparameters of DNN.

- **Data**:
performance datasets of 9 subject systems as specified in the paper.

- **Raw_results**:
contains the raw experiment results for all the research questions.

- **Figure5/6/7_full.pdf**:
supplementary tables for Figure 5/6/7 in the paper.

- **Requirements.txt**:
the required packages for running SeMPL_main.py.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python and Tensorflow:
the codes have been tested with **Python 3.6 - 3.7** and **Tensorflow 2.x**, other versions might cause errors.

3. Run *SeMPL_main.py* and install all missing packages according to runtime messages.


# Run *SeMPL*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the processes will be fully automated.

        python SeMPL_main.py
        
- **Python IDE (e.g. Pycharm)**: Open the *SeMPL_main.py* file on the IDE, and simply click 'Run'.


# Demo Experiment
The main program *SeMPL_main.py* defaultly runs a demo experiment that evaluates *SeMPL* with 5 sample sizes of *ImageMagick*, 
each repeated 30 times, without hyperparameter tuning (to save demonstration time).

A **succussful run** would produce similar messages as below: 

        Dataset: imagemagick-4tasks
        Number of expriments:  3
        Total sample size: 100, Number of features: 5
        Training sizes: [11, 24, 45, 66, 70], selected_tasks: [1]
        --- Evaluating imagemagick with S_1 ---
        Training size: 11, testing size: 89, Meta-training size: 100
        > Running sequence selection...
        > Meta-training [3, 1, 2] for target task 1...
            >> Learning task 3...
            	>> Learning task 1...
            >> Learning task 2...
        > Fine-tuning...
            >> Run1 imagemagick-4tasks S_1 T_1 MRE: 5.32, Training time (min): 0.02
            >> Run2 imagemagick-4tasks S_1 T_1 MRE: 4.29, Training time (min): 0.03
            >> Run3 imagemagick-4tasks S_1 T_1 MRE: 5.69, Training time (min): 0.03

The results will be saved in a file at the same directory with name in the format *'System_MainTask_MetaModel_FineTuningSamples-MetaSamples_Date'*, for example *'SeMPL_imagemagick-4tasks_T0_M[3, 1, 2]_11-100_03-28'*.

# Change Experiment Settings
To run more complicated experiments, alter the codes following the the instructions below and comments in *SeMPL_main.py*.

#### To switch between subject systems
    Comment and Uncomment the lines 33-40 following the comments in SeMPL_main.py.

    E.g., to run DaL with Apache, uncomment line 33 'subject_system = 'Apache_AllNumeric'' and comment out the other lines.


#### To save the experiment results
    Set 'save_file = True' at line 21.
    
    
#### To tune the hyperparameters (takes longer time)
    Set line 20 with 'test_mode = False'.


#### To change the number of experiments for specified sample size(s)
    Change 'N_experiments' at line 27, where each element corresponds a sample size. 

    For example, to simply run the first sample size with 30 repeated runs, set 'N_experiments = [30, 0, 0, 0, 0]'.

#### To change the sample sizes of a particular system
    Edit lines 55-71.

    For example, to run Apache with sample sizes 10, 20, 30, 40 and 50: set line 55 with 'sample_sizes = [10, 20, 30, 40, 50]'.


#### To compare DaL with DeepPerf
    1. Set line 20 with 'test_mode = False'.

    2. Set line 23 with 'enable_deepperf = True'.


#### To compare DaL with other ML models (RF, DT, LR, SVR, KRR, kNN) and DaL framework with these models (DaL_RF, DaL_DT, DaL_LR, DaL_SVR, DaL_KRR, DaL_kNN)
    1. Set line 20 with 'test_mode = False'.

    2. Set line 22 with 'enable_baseline_models = True'.


#### To run DaL with different depth d
    Add the dedicated d into the list 'depths' at line 25.
    
    E.g, run DaL with d=2: set 'depths = [2]'.

    E.g, run DaL with d=3 and d=4, respectively: set 'depths = [3, 4]'.


# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with *DaL* in the paper. 

- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.

- [DECART](https://github.com/jmguo/DECART)

    CART with data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature seleaction.

- [Perf-AL](https://github.com/GANPerf/GANPerf)

    Novel GAN based performance model with a generator to predict performance and a discriminator to distinguish the actual and predicted labels.
    


Note that *SeMPL_main.py* only compares *DeepPerf* because it is formulated in the most similar way to *DaL*, while the others are developed under different programming languages or have differnt ways of usage. 

Therefore, to compare *DaL* other SOTA models, please refer to their original pages (you might have to modify or reproduce their codes to ensure the compared models share the same set of training and testing samples).
