import os
import json
import sys
import pandas as pd
from IPython.display import display
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu

sys.path.insert(0, "/home/cloud/Projects/CAN/detect/") # add detect folder to path so that files from this folder can be imported
import signal_based_preprocess_functions

def get_captures_names():

    training_captures = [directory for directory in os.listdir("/home/cloud/Projects/CAN/actt/data-cancaptures/") if ("road_ambient_dyno" in directory) or ("road_ambient_highway" in directory)]
    testing_captures = [directory for directory in os.listdir("/home/cloud/Projects/CAN/actt/data-cancaptures/") if ("road_attack_" in directory) and not ("accelerator" in directory)]

    # print(len(training_captures), training_captures, "\n")   
    # print(len(testing_captures), testing_captures, "\n")   

    return training_captures, testing_captures


def get_metadata_info():

    with open("/home/cloud/Projects/CAN/actt/data/capture_metadata.json") as f:
        attack_metadata = json.load(f)

    attack_metadata_keys = [name for name in attack_metadata.keys() if "masquerade" in name and "accelerator" not in name]

    # print(len(attack_metadata_keys), list(attack_metadata_keys))

    return attack_metadata, attack_metadata_keys
    
def get_dbc_path():

    actt_path = os.path.join(os.path.join(os.path.expanduser("~"), "Projects", "CAN", "actt"))
    ground_truth_dbc_path = os.path.join(actt_path, "metadata", "dbcs", "heuristic_labeled", "anonymized_020822_030640.dbc")

    return ground_truth_dbc_path


def from_capture_to_time_series(cap, ground_truth_dbc_path, freq):
    
    signal_multivar_ts, timepts, aid_signal_tups = signal_based_preprocess_functions.capture_to_mv_signal_timeseries(cap, ground_truth_dbc_path, min_hz_msgs=freq)

    return signal_multivar_ts, timepts, aid_signal_tups


def process_multivariate_signals(signal_multivar_ts, aid_signal_tups, window_length, offset):
    
    # First dataframe
    # Convert matrix of time series into a dataframe
    df = pd.DataFrame({f"{tup[0]}_{tup[1]}": signal_multivar_ts[:,index] for index, tup in enumerate(aid_signal_tups)})
    # display(df)

    # Remove columns with constant values
    df = df.loc[:, (df != df.iloc[0]).any()] 
    # display(df)
    
    # Stadarization
    # df_standardized = (df-df.mean())/df.std()
    # display(df_standardized)

    # normalization
    df_standardized = (df-df.min())/(df.max()-df.min())
    
    # Partition of data frames
    n = df_standardized.shape[0]
    i = 0
    partition = []
    
    while (i + window_length) < n:
        partition.append(df_standardized.iloc[i:i + window_length, :])
        i = i + offset
        
    if i != n:
        partition.append(df_standardized.iloc[i:n, :])
        
    return partition


def compute_correlation_matrices(partition):
    
    corr_matrices = []

    for df in partition:

        # Remove columns with constant values
        df = df.loc[:, (df != df.iloc[0]).any()] 

        # Compute correlation matrix
        corr_matrices.append(df.corr(method="pearson"))
        # corr_matrices.append(np.corrcoef(df.to_numpy(), rowvar=False))
        
    return corr_matrices


def upper(df):
    """Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    """
    try:
        assert(type(df) == np.ndarray)
    except:
        if type(df) == pd.DataFrame:
            df = df.values
        else:
            raise TypeError("Must be np.ndarray or pd.DataFrame")
    mask = np.triu_indices(df.shape[0], k=1)
    
    return df[mask]


def compute_correlation_distribution_training(training_capture_name, ground_truth_dbc_path, freq):

    signal_multivar_ts, timepts, aid_signal_tups = from_capture_to_time_series(training_capture_name, ground_truth_dbc_path, freq)

    window = signal_multivar_ts.shape[0]
    offset = window
    partition_training = process_multivariate_signals(signal_multivar_ts, aid_signal_tups, window, offset)

    # print(len(partition_training))
    # display(partition_training[0])

    corr_matrices_training = compute_correlation_matrices(partition_training)
    signals_training = corr_matrices_training[0].columns.values

    corr_sample_training = np.concatenate([upper(corr_matrices_training[i]) for i in range(len(corr_matrices_training))])

    return(corr_sample_training, signals_training)



def compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset):

    signal_multivar_ts, timepts, aid_signal_tups = from_capture_to_time_series(testing_capture_name, ground_truth_dbc_path, freq)

    partition_testing = process_multivariate_signals(signal_multivar_ts, aid_signal_tups, window, offset)

    # print(len(partition_testing))
    # display(partition_testing[0])

    corr_matrices_testing = compute_correlation_matrices(partition_testing)

    # corr_sample_testing = upper(corr_matrices_testing[index_interval])

    return(corr_matrices_testing, timepts)



def create_time_intervals(total_length, window, offset):
    
    # Partition of data frames
    i = 0
    intervals = []
    
    while (i + window) < total_length:
        intervals.append((i, i + window))
        i = i + offset
        
    if i != total_length:
        intervals.append((i , total_length))
        
    return intervals



def process_testing_capture_correlation_ROAD(testing_capture_name, ground_truth_dbc_path, freq, window, 
                                             offset, corr_sample_training, injection_interval):

    corr_matrices_testing, timepts = compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset)
    total_length = timepts[-1]
    print("corr_matrices_testing: ", len(corr_matrices_testing))

    intervals_testing = create_time_intervals(total_length, window/freq, offset/freq)
    # print(intervals_testing[0:10])
    print("interval_testing: ", len(intervals_testing))

    tp, fp, fn, tn = 0, 0, 0, 0

    ground_truth = []
    predict_proba = []

    for index_interval in tqdm(range(len(intervals_testing))):
        
        # # Compute signal names intersection
        # signals_testing = corr_matrices_testing[index_interval].columns.values
        # signal_names_intersection = list(set(signals_training).intersection(set(signals_testing)))

        # # print(signal_names_intersection)
        
        # # Filter correlation matrices by common names
        # corr_sample_training = upper(corr_matrices_training[0].loc[signal_names_intersection, signal_names_intersection])
        # corr_sample_testing = upper(corr_matrices_testing[index_interval].loc[signal_names_intersection, signal_names_intersection])

        # Get correlation matrices
        corr_sample_testing = upper(corr_matrices_testing[index_interval])
        
        # Do hypothesis test
        mannwhitneyu_test = 1 - mannwhitneyu(corr_sample_training, corr_sample_testing)[1]

        # Evaluation criteria
        if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
            or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
            ground_truth.append(1)
            predict_proba.append(mannwhitneyu_test)
        else:
            ground_truth.append(0)
            predict_proba.append(mannwhitneyu_test)

    return ground_truth, predict_proba

        


