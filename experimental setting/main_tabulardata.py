# Support libs
import os
import time
import math
import warnings
import numpy as np
import pandas
import sklearn.model_selection as ms
from sklearn.exceptions import ConvergenceWarning
from multiprocessing import Process, Lock

# Models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Calibration
from netcal.binning import HistogramBinning, BBQ
from sklearn.calibration import CalibratedClassifierCV

# Results analysys
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from ReliabilityDiagram import ReliabilityDiagram # modified from the netcal version to resolve some bugs 
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve

# Used to save a classifier and measure its size in KB
from joblib import dump

# ------- GLOBAL VARS -----------

# Name of the folder in which look for tabular (CSV) datasets
CSV_FOLDER = "HW_Failure"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Percantage of test data wrt total data
TEST_SPLIT = 0.15
# Percantage of train data wrt total data
TRAIN_SPLIT = 0.4
# Percantage of calibration data wrt total data
CALIBRATION_SPLIT = 0.3
# True if you want to see the unique confidences of the models
SHOW_CONFIDENCE = False
# True if you want to calculate value
CALCULATE_VALUE = False
# True if debug information needs to be shown
VERBOSE = True
# List of random states the program takes
RANDOM_STATE = [42, 938465, 652894, 134875, 394856, 596657, 657492, 938563, 678430, 578231]
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "scores"
# Number of bins for functions that need them
N_BINS = 10
# Threshold for Histogram Binning and BBQ
THRESHOLD = 0.5
# Whether you want the experiments to run on multiple processes, the prints may be inconsistent, each random state is a single process
MULTIPROCESS = True
# Choose the DPI of the graphs you want to print
DPI = 300


# --------- SUPPORT FUNCTIONS ---------------

def current_milli_time():
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)

def get_learners(random_state):
    """
    Function to get a learner to use, given its string tag
    :param cont_perc: percentage of anomalies in the training set, required for unsupervised classifiers from PYOD
    :return: the list of classifiers to be trained
    """
    learners = [
        DecisionTreeClassifier(random_state=random_state),
        RandomForestClassifier(random_state=random_state),
        XGBClassifier(random_state=random_state),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(n_neighbors=1),
        LogisticRegression(random_state=random_state,max_iter=100000),
        # MLPClassifier(solver='lbfgs',
        #               alpha=1e-5,
        #               hidden_layer_sizes=(10),
        #               random_state=random_state,
        #               max_iter=100000),
        VotingClassifier(estimators=[('xgb', XGBClassifier(random_state=random_state)),
                                     #('lda', LinearDiscriminantAnalysis()),
                                     ('rf', RandomForestClassifier(random_state=random_state))],
                         voting='soft'),
    ]
    return learners

def plot_roc_curves(classifiers_dict, y_true, dataset_name, classifier_name, random_state):
    metrics_dir = os.path.join('.', f'graphs_{random_state}', dataset_name, 'roc')
    os.makedirs(metrics_dir, exist_ok=True)
    plt.figure()

    for classifier_names, y_proba in classifiers_dict.items():
        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
        gini = abs(metrics.roc_auc_score(y_true, y_proba)*2-1)
        plt.plot(fpr, tpr, label=f'{classifier_names} (Gini = {gini:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC {random_state} {dataset_name} {classifier_name}', fontsize=10)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(metrics_dir,f'{dataset_name}_{classifier_name}_roc.png'), format="png", dpi=DPI)
    plt.close()

def calculate_agf(TP, FP, TN, FN):
    # calculate F2 score, divide by 0 is considered perfect
    if(TP + FN==0):
        sens = 1
    else:
        sens = (TP) / (TP + FN)
    if(TP+FP==0):
        prec = 1
    else:
        prec = (TP) / (TP + FP)
        
    if (sens+prec==0):
        F2 = 0
    else:
        F2 = 5 * sens*prec / (4 * sens + prec)
    
    #swap labels of the truth matrix
    temp = FP
    TP = TN
    FP = FN
    FN = temp
    
    # calculate inverse F0_5 score, divide by 0 is considered perfect
    if(TP + FN==0):
        sens = 1
    else:
        sens = (TP) / (TP + FN)
    if(TP+FP==0):
        prec = 1
    else:
        prec = (TP) / (TP + FP)
        
    if (sens+prec==0):
        inv_F0_5 = 0
    else:
        inv_F0_5 = (5 / 4) * ((sens * prec) / (0.5 * 0.5 * sens + prec))
    
    return math.sqrt(F2 * inv_F0_5)

def print_unique_and_counts(variable_name, arr):
    # Get unique values and their counts
    unique_values, counts = np.unique(np.round(arr, 6), return_counts=True)
    
    # Initialize an empty string to hold the formatted output
    output_str = ""
    temp=0
    # Iterate over the unique values and their counts
    for val, count in zip(unique_values, counts):
        # Convert the value to a string and format it with the count
        output_str += f"{val:.6f} ({count:>{5}}) "
        temp+=1
        if (temp==6):
            #print("ciao")
            output_str += "\n"
            temp=0
    
    # Remove the trailing space and print the result
    print(f"{variable_name}: \n{output_str.rstrip()}")

# Value Aware functions

def cost_based_threshold(k):
    """calculate theoretical thresold

    Args:
        k (float):  how many times higher the abs value of an error is compared to a correct answer

    Returns:
        float: returns the optimal thresold given the cost
    """
    t = (k) / (k + 1)
    return t

def calculate_value(y_hat_proba, y, t_fp, V_fp, t_fn, V_fn, Vc, Vr):
    """ calculate value of classificator

    Args:
        y_hat_proba (2D npy array of float): contains confidences score on the set
        y (1D npy array of 0 or 1): contains ground truth of the set
        t_fp (float): thresold for false positive
        V_fp (float): value of FP
        t_fn (float): thresold for false negative
        V_fn (float): value of FN
        Vc (float): value of correct classification
        Vr (float): value of reject classification

    Returns:
        float: value of classificator 
        int: number of rejected samples
        int: number of wrong predictions
        int: number of correct predictions

    """

    values = [Vc, V_fp, V_fn]
    n_samples = len(y)
    value_vector = np.full(n_samples, Vr)

    # if any threshold is below 0.5 we need to make an extra check to assure that we are considering the most confident prediction
    if ((t_fp < 0.5) or (t_fn < 0.5)):
        # conditions to decide the value of the prediction
        cond1 = (((y == 1) & (y_hat_proba[:, 1] > t_fp) & (y_hat_proba[:, 1] > y_hat_proba[:, 0])) | ((y == 0) & (y_hat_proba[:, 0] > t_fn)) & (y_hat_proba[:, 0] > y_hat_proba[:, 1]))
        cond2 = (y_hat_proba[:, 1] > y_hat_proba[:, 0]) & (y != 1) & (y_hat_proba[:, 1] > t_fp)
        cond3 = (y_hat_proba[:, 0] > y_hat_proba[:, 1]) & (y != 0) & (y_hat_proba[:, 0] > t_fn)

        # Assigns the correct value to each prediction
        value_vector[cond1] = values[0]
        value_vector[cond2] = values[1]
        value_vector[cond3] = values[2]

    else:
        # conditions to decide the value of the prediction
        cond1 = ((y == 1) & (y_hat_proba[:, 1] > t_fp)) | ((y == 0) & (y_hat_proba[:, 0] > t_fn))
        cond2 = (y != 1) & (y_hat_proba[:, 1] > t_fp)
        cond3 = (y != 0) & (y_hat_proba[:, 0] > t_fn)

        # Assigns the correct value to each prediction
        value_vector[cond1] = values[0]
        value_vector[cond2] = values[1]
        value_vector[cond3] = values[2]

    # Calculate the total value
    value = np.sum(value_vector) / n_samples

    # Calculate the number of rejected samples, wrong predictions, and correct predictions
    numOfWrongPredictions = len(value_vector[cond2])+len(value_vector[cond3])
    numOfCorrectPredictions = len(value_vector[cond1])
    numOfRejectedSamples = n_samples - numOfCorrectPredictions - numOfWrongPredictions

    return value, numOfRejectedSamples, numOfWrongPredictions, numOfCorrectPredictions

def calculate_value_without_rejection(y_hat_proba, y, V_fp, V_fn, Vc):
    """ calculate value of classificator assuming rejection is not allowed

    Args:
        y_hat_proba (2D npy array of float): contains confidences score on the set
        y (1D npy array of 0 or 1): contains ground truth of the set
        V_fp (float): value of FP
        V_fn (float): value of FN
        Vc (float): value of correct classification

    Returns:
        float: value of classificator 
        int: number of wrong samples
        int: number of correct predictions

    """
    values = [V_fp, V_fn]
    n_samples = len(y)
    value_vector = np.full(n_samples, Vc)

    # conditions to decide the value of the prediction
    cond1 = (y != 1) & (y_hat_proba[:, 1] > y_hat_proba[:, 0])
    cond2 = (y != 0) & (y_hat_proba[:, 0] > y_hat_proba[:, 1])

    # Assigns the correct value to each prediction
    value_vector[cond1] = values[0]
    value_vector[cond2] = values[1]
    
    # Calculate the total value
    value = np.sum(value_vector) / n_samples

    # Calculate the number of wrong predictions and correct predictions
    numOfWrongPredictions = len(value_vector[cond1])+len(value_vector[cond2])
    numOfCorrectPredictions = n_samples-numOfWrongPredictions

    return value, numOfWrongPredictions, numOfCorrectPredictions

def find_optimum_confidence_threshold_fp(y_hat_proba, y, theoretical, Vw_fp, Vc, Vr, confidence, precision):
    """ calculates the best empirical t_fp given a precision and a confidence

    Args:
        y_hat_proba (2D npy array of float): contains confidences score on the set
        y (1D npy array of 0 or 1): contains ground truth of the set
        theoretical (float): contains the middle point of the search, good estimate would be the 
                             theoretical threshold
        Vw_fp (float): value of FP
        Vc (float): value of correct classification
        confidence (float): how far from the middle point to search, 1 is equal to searching the whole [0,1] range,
                            should be positive
        precision (int): number of decimal numbers of the step to iterate inside the [0,1] range, 
                         1 means a step of 0.1, should be positive 
    Returns:
        float: best empirical t_fp given a precision and a confidence

    """
    #initialize
    max_value=float('-inf')
    max_t_fp = 0
    step = 10 ** -precision
    # rounds to be consistent
    theoretical=round(theoretical,precision)
    # iterates over all values of t_fp within a confidence range of the theoretical threshold with a step
    for t_fp in np.arange(max(theoretical-confidence,0), min(theoretical+confidence+step,1+step), step):
        value,_,_,_ = calculate_value(y_hat_proba, y, t_fp, Vw_fp, 0.6, 0, Vc, Vr)
        #if value is higher select new best threshold
        if(max_value<value):
            max_value=value
            max_t_fp=t_fp
    return max_t_fp

def find_optimum_confidence_threshold_fn(y_hat_proba, y, theoretical, Vw_fn, Vc, Vr, confidence, precision):
    """ calculates the best empirical t_fn given a precision and a confidence

    Args:
        y_hat_proba (2D npy array of float): contains confidences score on the set
        y (1D npy array of 0 or 1): contains ground truth of the set
        theoretical (float): contains the middle point of the search, good estimate would be the 
                             theoretical threshold
        Vw_fn (float): value of FN
        Vc (float): value of correct classification
        confidence (float): how far from the middle point to search, 1 is equal to searching the whole [0,1] range,
                            should be positive
        precision (int): number of decimal numbers of the step to iterate inside the [0,1] range, 
                         1 means a step of 0.1, should be positive 
    Returns:
        float: best empirical t_fn given a precision and a confidence

    """
    #initialize
    max_value=float('-inf')
    max_t_fn=0
    step = 10 ** -precision
    # rounds to be consistent
    theoretical=round(theoretical,precision)
    # iterates over all values of t_fn within a confidence range of the theoretical threshold with a step
    for t_fn in np.arange(max(theoretical-confidence,0), min(theoretical+confidence+step,1+step), step):
        value,_,_,_ = calculate_value(y_hat_proba, y, 0.6, 0, t_fn, Vw_fn, Vc, Vr)
        #if value is higher select new best threshold
        if(max_value<value):
            max_value=value
            max_t_fn=t_fn
    return max_t_fn

def cost_based_analysis(
    y_hat_proba_val,
    y_val,
    y_hat_proba_test,
    y_test,
    res_path,
    logfile_name,
    Vr,
    Vc,
    Vw_list_fp,
    Vw_list_fn,
    precision_fp,
    precision_fn,
    confidence_fp,
    confidence_fn,
):
    """ creates a file containing the value of a series of predictions using both a theoretical 
    and an empirically found perfect threshold for all the different error costs combinations 

    Args:
        y_hat_proba_val (2D npy array of float): contains confidences score on the validation set
        y_val (1D npy array of 0 or 1): contains ground truth of the validation set
        y_hat_proba_test (2D npy array of float): contains confidences score on the test set
        y_test (1D npy array of 0 or 1): contains ground truth of the test set
        res_path (str): directory in which to print the file
        logfile_name (str): name of result file
        Vc (float): value of correct classification
        Vr (float): value of reject classification
        Vw_list_fp (list of float): list of values of FP error
        Vw_list_fn (list of float): list of values of FN error
        precision_fp (int): precision to use when searching for empirical threshold fp
        precision_fn (int): precision to use when searching for empirical threshold fn
        confidence_fp (float): confidence to use when searching for empirical threshold fp
        confidence_fn (float): confidence to use when searching for empirical threshold fn
    """
    # create log file
    os.makedirs(os.path.join(os.getcwd(), res_path),exist_ok=True)
    rc_path = os.path.join(os.getcwd(), res_path, logfile_name + "_costBased_test.csv")
    with open(rc_path, "w") as f:
        c = "Vr,Vc,Vw_fp,Vw_fn,k_fp,k_fn,t_fp,t_fn,value,rejected,wrong,correct,t_optimal_fp,t_optimal_fn,value_optimal,rejected_opt,wrong_opt,correct_opt,value_no_rej,wrong_no_rej,correct_no_rej"
        f.write(c + "\n")

    # pre-compute all theoretical thresholds for each FN value
    fn_list=[]
    for Vw_fn in Vw_list_fn:
        # calculate theoretical threshold
        k_fn = (-1) * (Vw_fn / Vc)
        t_fn = cost_based_threshold(k_fn)
        # calculate empirically perfect threshold
        fn_list.append((find_optimum_confidence_threshold_fn(
            y_hat_proba_val, y_val, t_fn, Vw_fn, Vc, Vr, confidence_fn, precision_fn
        ),t_fn,Vw_fn))

    # iterates over all possible values for FP
    for Vw_fp in Vw_list_fp:
        data_log = []
        
        # calculate theoretical threshold
        k_fp = (-1) * (Vw_fp / Vc)
        t_fp = cost_based_threshold(k_fp)
        
        # calculate empirically perfect threshold
        e_fp = find_optimum_confidence_threshold_fp(
                y_hat_proba_val, y_val, t_fp, Vw_fp, Vc, Vr, confidence_fp, precision_fp
            )
        
        for e_fn,t_fn,Vw_fn in fn_list:
            #calculate value using theoretical best threshold
            value_test, rej_test, wrong_test, correct_test = calculate_value(
                y_hat_proba_test, y_test, e_fp, Vw_fp, t_fn, Vw_fn, Vc, Vr
            )
            
            #calculate value using empirical perfect threshold
            value_test_opt, rej_test_opt, wrong_test_opt, correct_test_opt = calculate_value(
                y_hat_proba_test,
                y_test,
                e_fp,
                Vw_fp,
                e_fn,
                Vw_fn,
                Vc,
                Vr,
            )

            # calculate value assuming no rejection
            value_test_no_rej, wrong_test_no_rej, correct_test_no_rej = calculate_value_without_rejection(
                y_hat_proba_test, y_test, Vw_fp, Vw_fn, Vc
            )
            
            # calculate theoretical threshold
            k_fn = (-1) * (Vw_fn / Vc)
            
            # handles output to file
            data_log.append(
                f"{Vr},{Vc},{Vw_fp},{Vw_fn},{k_fp},{k_fn},{t_fp},{t_fn},{value_test},{rej_test},{wrong_test},{correct_test},{e_fp},{e_fn},{value_test_opt},{rej_test_opt},{wrong_test_opt},{correct_test_opt},{value_test_no_rej},{wrong_test_no_rej},{correct_test_no_rej}\n"
            )
        
        with open(rc_path, "a") as f:
            for i in data_log:
                f.write(i)

# ----------------------- MAIN ROUTINE ---------------------

def single_run(lock,random_state):
    np.set_printoptions(suppress=True)
    #Initiate metrics
    ece = ECE(N_BINS)

    #value aware setup
    Vr = 0.0
    Vc = 1.0

    Vw_list_fn = list(np.arange(-10.0, -1000.1, -10.0))
    Vw_list_fp = [-10.0]

    resPath = "value"

    # we ignore the warnings of failure of convergence as LogisticRegression and MLP rarely converge outside the last dataset
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    total_time = current_milli_time()

    with open(f'{SCORES_FILE}_{random_state}.csv', 'w') as f:
        f.write("dataset_tag,classificator,calibrator,test_split,train_split,cal_split,emp_split,acc,mcc,ece,f1,agf,gini,gmean,time,model_size,")
        f.write('TN,TP,FN,FP')
        f.write("\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):
            # if file is a CSV, it is assumed to be a dataset to be processed
            lock.acquire()
            try:
                df = pandas.read_csv(full_name, sep=",")
            finally:
                lock.release()

            if VERBOSE:
                print("\n------------ DATASET INFO -----------------")
                print("Data Points in Dataset '%s': %d" % (dataset_file, len(df.index)))
                print("Features in Dataset: " + str(len(df.columns)))

            # Filling NaN and Handling (Removing) constant features
            df = df.fillna(0)
            df = df.loc[:, df.nunique() > 1]
            if VERBOSE:
                print("Features in Dataframe after removing constant ones: " + str(len(df.columns)))

            features_no_cat = df.select_dtypes(exclude=['object']).columns
            if VERBOSE:
                print("Features in Dataframe after non-numeric ones (including label): " + str(len(features_no_cat)))

            normal_perc = None
            y = df[LABEL_NAME].to_numpy()
            y = np.where(df[LABEL_NAME] == "normal", 0, 1)
            if VERBOSE:
                normal_frame = df.loc[df[LABEL_NAME] == NORMAL_TAG]
                normal_perc = len(normal_frame.index) / len(df.index)
                print("Normal data: " + str(len(normal_frame.index)) + " items (" +
                        "{:.3f}".format(100.0 * normal_perc) + "%)")
                
            # Set up subsets excluding categorical values that some algorithms cannot handle
            # 1-Hot-Encoding or other approaches may be used instead of removing
            x_no_cat = df.select_dtypes(exclude=['object']).to_numpy()
            x_train, x_temp, y_train, y_temp = ms.train_test_split(x_no_cat, y, train_size=TRAIN_SPLIT, shuffle=True, stratify=y, random_state=random_state)
            temp_split=1-TRAIN_SPLIT
            x_cal, x_temp, y_cal, y_temp = ms.train_test_split(x_temp, y_temp, train_size=CALIBRATION_SPLIT/temp_split, shuffle=True, stratify=y_temp, random_state=random_state)
            temp_split=1-CALIBRATION_SPLIT-TRAIN_SPLIT
            x_test, x_emp, y_test, y_emp = ms.train_test_split(x_temp, y_temp, train_size=TEST_SPLIT/temp_split, shuffle=True, stratify=y_temp, random_state=random_state)
            
            if VERBOSE:
                print(f"train set: {len(y_train)} or {TRAIN_SPLIT*100}%, calibration set: {len(y_cal)} or {CALIBRATION_SPLIT*100}%, test set: {len(y_test)} or {TEST_SPLIT*100}% and empirical set: {len(y_emp)} or {(1-TRAIN_SPLIT-CALIBRATION_SPLIT-TEST_SPLIT)*100}%")
                print('-------------------- CLASSIFIERS -----------------------')

            # Setups directory to save ReliabilityDiagrams
            rela_dir = os.path.join('.', f'graphs_{random_state}', dataset_file, 'ReliabilityDiagram')
            os.makedirs(rela_dir, exist_ok=True)
            
            # Loop for training and testing each learner specified by LEARNER_TAGS
            for classifier in get_learners(random_state):
                roc_dict = {}
                # Training the algorithm to get a model
                start_time = current_milli_time()
                classifier.fit(x_train, y_train)

                # Quantifying size of the model
                dump(classifier, f"clf_dump_{random_state}.bin")
                size = os.stat(f"clf_dump_{random_state}.bin").st_size
                os.remove(f"clf_dump_{random_state}.bin")

                # Base model
                y_pred = classifier.predict(x_test)
                y_conf = classifier.predict_proba(x_test)
                y_conf_cal = classifier.predict_proba(x_cal)
                acc = metrics.accuracy_score(y_test, y_pred)
                mcc = abs(metrics.matthews_corrcoef(y_test, y_pred))
                ece_score = ece.measure(y_conf, y_test)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
                f1 = metrics.f1_score(y_test, y_pred)
                agf = calculate_agf(tp,fp,tn,fn)
                gini = abs(metrics.roc_auc_score(y_test,y_conf[:,1])*2-1)
                if(tp + fn==0):
                    sens = 1
                else:
                    sens = (tp) / (tp + fn)
                if(tn+fp==0):
                    spec = 1
                else:
                    spec = (tn) / (tn + fp)
                gmean = math.sqrt(sens*spec)
                diagram = ReliabilityDiagram(N_BINS, title_suffix=f"{random_state} - {dataset_file} - {classifier.__class__.__name__} - BASE")
                diagram.plot(y_conf, y_test).savefig(os.path.join(rela_dir, f'{classifier.__class__.__name__}_Base.png'), format="png", dpi=DPI)
                plt.close()
                roc_dict[classifier.__class__.__name__] = y_conf[:,1]
                # base time of training model to be added to the time taken to calibrate
                base_time = current_milli_time() - start_time
                with open(f'{SCORES_FILE}_{random_state}.csv', "a") as myfile:
                    myfile.write(dataset_file + "," + classifier.__class__.__name__ + ",Base," +
                                str(TEST_SPLIT) + ',' + str(TRAIN_SPLIT) + ',' + str(CALIBRATION_SPLIT) + ',' + 
                                str(1-TEST_SPLIT-TRAIN_SPLIT-CALIBRATION_SPLIT) + ',' +
                                str(acc) + "," + str(mcc) + "," + str(ece_score) + "," +
                                str(f1) + "," + str(agf) + "," + str(gini) + "," + str(gmean) + "," +
                                str(base_time) + "," + str(size)+ ",")
                    myfile.write(f'{tn},{tp},{fn},{fp}')
                    myfile.write("\n")

                if CALCULATE_VALUE:
                    # value aware experiments
                    logfile_name = f'{dataset_file}_{classifier.__class__.__name__}_{random_state}_base'
                    y_conf_emp=classifier.predict_proba(x_emp)
                    cost_based_analysis(y_conf_emp, y_emp, y_conf, y_test, resPath, 
                                        logfile_name, Vr, Vc, Vw_list_fp, Vw_list_fn, 3, 3, 1, 1)

                # Platt Scaling Calibration
                start_time=current_milli_time()
                platt = CalibratedClassifierCV(classifier, cv='prefit')
                platt.fit(x_cal, y_cal)
                y_pred_pla = platt.predict(x_test)
                y_conf_pla = platt.predict_proba(x_test)
                acc_pla = metrics.accuracy_score(y_test, y_pred_pla)
                mcc_pla = abs(metrics.matthews_corrcoef(y_test, y_pred_pla))
                ece_score_pla = ece.measure(y_conf_pla, y_test)
                diagram = ReliabilityDiagram(N_BINS, title_suffix=f"{random_state} - {dataset_file} - {classifier.__class__.__name__} - PLA")
                diagram.plot(y_conf_pla, y_test).savefig(os.path.join(rela_dir, f'{classifier.__class__.__name__}_Platt.png'), format="png", dpi=DPI)
                plt.close()
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_pla).ravel()
                f1_pla = metrics.f1_score(y_test, y_pred_pla)
                agf_pla = calculate_agf(tp,fp,tn,fn)
                gini_pla = abs(metrics.roc_auc_score(y_test,y_conf_pla[:,1])*2-1)
                if(tp + fn==0):
                    sens = 1
                else:
                    sens = (tp) / (tp + fn)
                if(tn+fp==0):
                    spec = 1
                else:
                    spec = (tn) / (tn + fp)
                gmean_pla = math.sqrt(sens*spec)
                roc_dict[f'{classifier.__class__.__name__}_pla'] = y_conf_pla[:,1]

                with open(f'{SCORES_FILE}_{random_state}.csv', "a") as myfile:
                    myfile.write(dataset_file + "," + classifier.__class__.__name__ + ",Platt," +
                                str(TEST_SPLIT) + ',' + str(TRAIN_SPLIT) + ',' + str(CALIBRATION_SPLIT) + ',' +
                                str(1-TEST_SPLIT-TRAIN_SPLIT-CALIBRATION_SPLIT) + ',' +
                                str(acc_pla) + "," + str(mcc_pla) + "," + str(ece_score_pla) + "," +
                                str(f1_pla) + "," + str(agf_pla) + "," + str(gini_pla) + "," + str(gmean_pla) + "," +
                                str(current_milli_time() - start_time + base_time) + "," + str(size)+ ",")
                    myfile.write(f'{tn},{tp},{fn},{fp}')
                    myfile.write("\n")
                
                if CALCULATE_VALUE:
                    # value aware experiments
                    logfile_name = f'{dataset_file}_{classifier.__class__.__name__}_{random_state}_pla'
                    y_conf_emp_pla=platt.predict_proba(x_emp)
                    cost_based_analysis(y_conf_emp_pla, y_emp, y_conf_pla, y_test, resPath,
                                        logfile_name, Vr, Vc, Vw_list_fp, Vw_list_fn, 3, 3, 1, 1)
                
                # Isotonic Scaling Calibration
                start_time=current_milli_time()
                isotonic = CalibratedClassifierCV(classifier, cv='prefit', method='isotonic')
                isotonic.fit(x_cal, y_cal)
                y_pred_iso = isotonic.predict(x_test)
                y_conf_iso = isotonic.predict_proba(x_test)
                acc_iso = metrics.accuracy_score(y_test, y_pred_iso)
                mcc_iso = abs(metrics.matthews_corrcoef(y_test, y_pred_iso))
                ece_score_iso = ece.measure(y_conf_iso, y_test)
                diagram = ReliabilityDiagram(N_BINS, title_suffix=f"{random_state} - {dataset_file} - {classifier.__class__.__name__} - iso")
                diagram.plot(y_conf_iso, y_test).savefig(os.path.join(rela_dir, f'{classifier.__class__.__name__}_Isotonic.png'), format="png", dpi=DPI)
                plt.close()
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_iso).ravel()
                f1_iso = metrics.f1_score(y_test, y_pred_iso)
                agf_iso = calculate_agf(tp, fp, tn, fn)
                gini_iso = abs(metrics.roc_auc_score(y_test, y_conf_iso[:, 1])*2-1)
                if(tp + fn==0):
                    sens = 1
                else:
                    sens = (tp) / (tp + fn)
                if(tn+fp==0):
                    spec = 1
                else:
                    spec = (tn) / (tn + fp)
                gmean_iso = math.sqrt(sens*spec)
                roc_dict[f'{classifier.__class__.__name__}_iso'] = y_conf_iso[:,1]
                
                with open(f'{SCORES_FILE}_{random_state}.csv', "a") as myfile:
                    myfile.write(dataset_file + "," + classifier.__class__.__name__ + ",Isotonic," +
                                str(TEST_SPLIT) + ',' + str(TRAIN_SPLIT) + ',' + str(CALIBRATION_SPLIT) + ',' +
                                str(1-TEST_SPLIT-TRAIN_SPLIT-CALIBRATION_SPLIT)  + ',' +
                                str(acc_iso) + "," + str(mcc_iso) + "," + str(ece_score_iso) + "," +
                                str(f1_iso) + "," + str(agf_iso) + "," + str(gini_iso) + "," + str(gmean_iso) + "," +
                                str(current_milli_time() - start_time + base_time) + "," + str(size) + ",")
                    myfile.write(f'{tn},{tp},{fn},{fp}')
                    myfile.write("\n")
                
                if CALCULATE_VALUE:
                    # value aware experiments
                    logfile_name = f'{dataset_file}_{classifier.__class__.__name__}_{random_state}_iso'
                    y_conf_emp_iso=isotonic.predict_proba(x_emp)
                    cost_based_analysis(y_conf_emp_iso, y_emp, y_conf_iso, y_test, resPath,
                                        logfile_name, Vr, Vc, Vw_list_fp, Vw_list_fn, 3, 3, 1, 1)

                # Histogram Binning Calibration
                start_time=current_milli_time()
                histogram_binning = HistogramBinning(bins=N_BINS)
                histogram_binning.fit(y_conf_cal[:,1], y_cal, random_state=random_state)
                y_conf_his = histogram_binning.transform(y_conf[:,1], random_state=random_state)
                y_pred_his = (y_conf_his > THRESHOLD).astype(int)
                acc_his = metrics.accuracy_score(y_test, y_pred_his)
                mcc_his = abs(metrics.matthews_corrcoef(y_test, y_pred_his))
                ece_score_his = ece.measure(y_conf_his, y_test)
                diagram = ReliabilityDiagram(N_BINS, title_suffix=f"{random_state} - {dataset_file} - {classifier.__class__.__name__} - HB")
                diagram.plot(y_conf_his, y_test).savefig(os.path.join(rela_dir, f'{classifier.__class__.__name__}_HistogramBinning.png'), format="png", dpi=DPI)
                plt.close()
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_his).ravel()
                f1_his = metrics.f1_score(y_test, y_pred_his)
                agf_his = calculate_agf(tp, fp, tn, fn)
                gini_his = abs(metrics.roc_auc_score(y_test, y_conf_his)*2-1)
                if(tp + fn==0):
                    sens = 1
                else:
                    sens = (tp) / (tp + fn)
                if(tn+fp==0):
                    spec = 1
                else:
                    spec = (tn) / (tn + fp)
                gmean_his = math.sqrt(sens*spec)
                roc_dict[f'{classifier.__class__.__name__}_his'] = y_conf_his

                with open(f'{SCORES_FILE}_{random_state}.csv', "a") as myfile:
                    myfile.write(dataset_file + "," + classifier.__class__.__name__ + ",HB,"  +
                                str(TEST_SPLIT) + ',' + str(TRAIN_SPLIT) + ',' + str(CALIBRATION_SPLIT) + ',' + 
                                str(1-TEST_SPLIT-TRAIN_SPLIT-CALIBRATION_SPLIT)  + ',' +
                                str(acc_his) + "," + str(mcc_his) + "," + str(ece_score_his) + "," +
                                str(f1_his) + "," + str(agf_his) + "," + str(gini_his) + "," + str(gmean_his) + "," +
                                str(current_milli_time() - start_time + base_time) + "," + str(size) + ",")
                    myfile.write(f'{tn},{tp},{fn},{fp}')
                    myfile.write("\n")

                if CALCULATE_VALUE:
                    # value aware experiments
                    logfile_name = f'{dataset_file}_{classifier.__class__.__name__}_{random_state}_his'
                    y_conf_emp_his=histogram_binning.transform(y_conf_emp[:,1], random_state=random_state)
                    y_conf_emp_his=np.column_stack((1-y_conf_emp_his,y_conf_emp_his))
                    y_conf_his_2d=np.column_stack((1-y_conf_his,y_conf_his))
                    cost_based_analysis(y_conf_emp_his, y_emp, y_conf_his_2d, y_test, resPath,
                                        logfile_name, Vr, Vc, Vw_list_fp, Vw_list_fn, 3, 3, 1, 1)
                
                # Bayesian Binning into Quantiles (BBQ) Calibration
                start_time=current_milli_time()
                bbq_calibration = BBQ()
                bbq_calibration.fit(y_conf_cal[:,1], y_cal)
                #have to clip due to bug in library
                y_conf_bbq = np.clip(bbq_calibration.transform(y_conf[:,1]),0,1)
                y_pred_bbq = (y_conf_bbq > THRESHOLD).astype(int)
                acc_bbq = metrics.accuracy_score(y_test, y_pred_bbq)
                mcc_bbq = abs(metrics.matthews_corrcoef(y_test, y_pred_bbq))
                ece_score_bbq = ece.measure(y_conf_bbq, y_test)
                diagram = ReliabilityDiagram(N_BINS, title_suffix=f"{random_state} - {dataset_file} - {classifier.__class__.__name__} - BBQ")
                diagram.plot(y_conf_bbq, y_test).savefig(os.path.join(rela_dir, f'{classifier.__class__.__name__}_BBQ.png'), format="png", dpi=DPI)
                plt.close()
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_bbq).ravel()
                f1_bbq = metrics.f1_score(y_test, y_pred_bbq)
                agf_bbq = calculate_agf(tp, fp, tn, fn)
                gini_bbq = abs(metrics.roc_auc_score(y_test, y_conf_bbq)*2-1)
                if(tp + fn==0):
                    sens = 1
                else:
                    sens = (tp) / (tp + fn)
                if(tn+fp==0):
                    spec = 1
                else:
                    spec = (tn) / (tn + fp)
                gmean_bbq = math.sqrt(sens*spec)
                roc_dict[f'{classifier.__class__.__name__}_bbq'] = y_conf_bbq

                with open(f'{SCORES_FILE}_{random_state}.csv', "a") as myfile:
                    myfile.write(dataset_file + "," + classifier.__class__.__name__ + ",BBQ," +
                                str(TEST_SPLIT) + ',' + str(TRAIN_SPLIT) + ',' + str(CALIBRATION_SPLIT) + ',' + 
                                str(1-TEST_SPLIT-TRAIN_SPLIT-CALIBRATION_SPLIT)  + ',' +
                                str(acc_bbq) + "," + str(mcc_bbq) + "," + str(ece_score_bbq) + "," +
                                str(f1_bbq) + "," + str(agf_bbq) + "," + str(gini_bbq) + "," + str(gmean_bbq) + "," +
                                str(current_milli_time() - start_time + base_time) + "," + str(size) + ",")
                    myfile.write(f'{tn},{tp},{fn},{fp}')
                    myfile.write("\n")

                if CALCULATE_VALUE:
                    # value aware experiments
                    logfile_name = f'{dataset_file}_{classifier.__class__.__name__}_{random_state}_bbq'
                    #have to clip due to bug in library
                    y_conf_emp_bbq=np.clip(bbq_calibration.transform(y_conf_emp[:,1]),0,1)
                    y_conf_emp_bbq=np.column_stack((1-y_conf_emp_bbq,y_conf_emp_bbq))
                    y_conf_bbq_2d=np.column_stack((1-y_conf_bbq,y_conf_bbq))
                    cost_based_analysis(y_conf_emp_bbq, y_emp, y_conf_bbq_2d, y_test, resPath,
                                        logfile_name, Vr, Vc, Vw_list_fp, Vw_list_fn, 3, 3, 1, 1)

                # Plot roc curve
                plot_roc_curves(roc_dict, y_test, dataset_file, classifier.__class__.__name__,random_state)

                # Plot calibration curve
                y_conf=y_conf[:, 1]
                y_conf_pla=y_conf_pla[:, 1]
                y_conf_iso=y_conf_iso[:, 1]
                fop_uncalibrated, mpv_uncalibrated = calibration_curve(y_test, y_conf, n_bins=N_BINS)
                fop_pla, mpv_pla = calibration_curve(y_test, y_conf_pla, n_bins=N_BINS)
                fop_iso, mpv_iso = calibration_curve(y_test, y_conf_iso, n_bins=N_BINS)
                fop_his, mpv_his = calibration_curve(y_test, y_conf_his, n_bins=N_BINS)
                fop_bbq, mpv_bbq = calibration_curve(y_test, y_conf_bbq, n_bins=N_BINS)

                plt.figure(figsize=(10, 5))
                plt.plot([0, 1], [0, 1], linestyle='--', color='black')
                plt.plot(mpv_uncalibrated, fop_uncalibrated, marker='.', label='Uncalibrated')
                plt.plot(mpv_pla, fop_pla, marker='.', label='Platt')
                plt.plot(mpv_iso, fop_iso, marker='.', label='Isotonic')
                plt.plot(mpv_his, fop_his, marker='.', label='Histogram Binning')
                plt.plot(mpv_bbq, fop_bbq, marker='.', label='BBQ')

                plt.xlabel('Mean Predicted Value')
                plt.ylabel('Fraction of Positives')
                plt.title(f'Calibration Curve for {classifier.__class__.__name__}_{dataset_file}')
                plt.legend()
                plt.savefig(os.path.join(rela_dir, f'{classifier.__class__.__name__}_Curve.png'), format="png", dpi=DPI)
                plt.close()
                plt.close('all')

                print(f'finished {classifier.__class__.__name__}')

                if SHOW_CONFIDENCE:        
                    print_unique_and_counts("base", y_conf)
                    print_unique_and_counts("platt", y_conf_pla)
                    print_unique_and_counts("iso", y_conf_iso)
                    print_unique_and_counts("his", y_conf_his)
                    print_unique_and_counts("BBQ", y_conf_bbq)
                
            
    print(f'Total run time of {random_state}: {(current_milli_time()-total_time) / 1000} seconds')

if __name__ == '__main__':
    lock = Lock()
    
    processes = []
   
    if MULTIPROCESS:
        for random_state in RANDOM_STATE:
            p = Process(target=single_run, args=(lock, random_state,))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        for random_state in RANDOM_STATE:
            p = Process(target=single_run, args=(lock, random_state,))
            p.start()
            p.join()

'''
Copyright 2024 Nicolò Marchini
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''