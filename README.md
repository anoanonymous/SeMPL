# One at a Time: Predicting Configuration Performance in Multiple Environments with Sequential Meta-Learning
>Learning and predicting the performance of given software configurations are of high importance to many software engineering activities. While configurable software systems will almost certainly face diverse running environments (e.g., version, hardware, and workload), current work often either build performance models under a single environment or fails to properly handle data from diverse settings, hence restricting their accuracy for a new environment.
>
>In this paper, we target configuration performance learning under multiple environments. We do so by designing **SeMPL** —— a meta-learning framework that learns the common understanding from configurations measured in distinct environments (meta-tasks) and generalizes them to the target setting (target-task). What makes it unique is that unlike common meta-learning frameworks (e.g., MAML and MetaSGD) that train the meta-tasks in parallel, we train them sequentially, one at a time. The order of training naturally allows discriminating the contributions among meta-tasks in the meta-model built, which fits better with the characteristic of configuration data that is known to dramatically differ between different environments/tasks. 
>
>Through comparing with 15 state-of-the-art models under 9 systems, our extensive experimental results demonstrate that *SeMPL* performs considerably better in **89%** of the cases with up to **99%** accuracy improvement, while being data-efficient, leading to at most **3.86×** speedup.

This repository contains the **key codes**, **full data used**, **raw experiment results** and **the supplementary tables** for the paper.

# Documents
- **Codes**

    └─ **SeMPL_main.py**: 
the *main program* for using SeMPL, which automatically reads data from csv files, trains and evaluates, and save the results.

    └─ **sequence_selection.py**: 
the key codes for selecting the best sequence for a given system.

    └─ **meta_training.py**: 
the key codes for selecting the best sequence for a given system.

    └─ **mlp_plain/sparse_model.py**:
contains functions to construct and train rDNN. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf). 

    └─ **Meta_plain/sparse_model.py**:
improved rDNN models that can be pre-trained with sequential meta-learning. 

    └─ **utils / general.py**:
    contains utility functions to build DNN and other ML models.
    
    └─ **utils / hyperparameter_tuning.py**:
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
        > Meta-training [3, 1, 2] for target task T_1...
            >> Learning task 3...
            	>> Learning task 1...
            >> Learning task 2...
        > Fine-tuning...
            >> Run1 imagemagick-4tasks S_1 T_1 MRE: 5.32, Training time (min): 0.02
            >> Run2 imagemagick-4tasks S_1 T_1 MRE: 4.29, Training time (min): 0.03
            >> Run3 imagemagick-4tasks S_1 T_1 MRE: 5.69, Training time (min): 0.03

The results will be saved in a file at the same directory with name in the format *'System_MainTask_MetaModel_FineTuningSamples-MetaSamples_Date'*, for example *'imagemagick-4tasks_T0_M[3, 1, 2]_11-100_03-28.txt'*.

# Change Experiment Settings
To run more complicated experiments, alter the codes following the the instructions below and comments in *SeMPL_main.py*.

#### To switch between subject systems
    Change the line 20 in SeMPL_main.py.

    E.g., to run SeMPL with DeepArch and SaC, simply write 'selected_sys = [0, 1]'.
    
    
#### To tune the hyperparameters (takes longer time)
    Set line 23 with 'test_mode = False'.


#### To change the number of experiments for specified sample size(s)
    Change 'N_experiments' at line 26.
    

# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with *SeMPL* in the paper. 

#### Single Environment Performance Models
- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.
    
- [RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

    A commonly used ensemble of trees that tackle the feature sparsity issue.

- [DECART](https://github.com/jmguo/DECART)

    A improved regression tree with data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature seleaction.
   
#### Joint Learning for Performance Models

- [BEETLE](https://github.com/ai-se/BEETLE)

   A model that selects the bellwether environment for transfer learning.
   
- [tEAMS](https://zenodo.org/record/4960172#.ZCHaK8JBzN8)

   A recent approach that reuses and transfers the performance model during software evolution.
   
- [MORF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

   A multi-task learning version of RF where there is one dedicated output for each task of performance prediction.
    
#### Meta-Learning Models

- [MAML](https://github.com/cbfinn/maml)

   A state-of-the-art meta-learning framework that has been widely applied in different domains, including software engineering.
   
- [MetaSGD](https://github.com/jik0730/Meta-SGD-pytorch)

   Extends the MAML by additionally adapting the learning rate along the meta-training process, achieving learning speedup over MAML


To compare *SeMPL* with other SOTA models, please refer to their original pages (you might have to modify or reproduce their codes to ensure the compared models share the same set of training and testing samples).
