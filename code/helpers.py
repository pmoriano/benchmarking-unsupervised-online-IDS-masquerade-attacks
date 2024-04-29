import os
import json
import sys
import pandas as pd
from IPython.display import display
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from scipy.stats.mstats import spearmanr
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.stats import norm
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram, linkage, fcluster
from clusim.clustering import Clustering, remap2match
import clusim.sim as sim
import time


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



def compute_correlation_training(training_capture_name, ground_truth_dbc_path, freq):

    signal_multivar_ts, timepts, aid_signal_tups = from_capture_to_time_series(training_capture_name, ground_truth_dbc_path, freq)

    window = signal_multivar_ts.shape[0]
    offset = window
    partition_training = process_multivariate_signals(signal_multivar_ts, aid_signal_tups, window, offset)

    # print(len(partition_training))
    # display(partition_training[0])

    corr_matrices_training = compute_correlation_matrices(partition_training)
    signals_training = corr_matrices_training[0].columns.values

    # corr_sample_training = np.concatenate([upper(corr_matrices_training[i]) for i in range(len(corr_matrices_training))])

    return(corr_matrices_training, signals_training)



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



def process_testing_capture_distribution_ROAD(testing_capture_name, ground_truth_dbc_path, freq, window, 
                                             offset, corr_sample_training, injection_interval):

    corr_matrices_testing, timepts = compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset)
    total_length = timepts[-1]
    # print("corr_matrices_testing: ", len(corr_matrices_testing))

    intervals_testing = create_time_intervals(total_length, window/freq, offset/freq)
    # print(intervals_testing[0:10])
    print("interval_testing: ", len(intervals_testing))

    ground_truth = []
    predict_proba = []

    start = time.time()

    for index_interval in range(len(intervals_testing)):
        
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

    end = time.time()

    ttw = (end - start)/len(intervals_testing)

    return ground_truth, predict_proba, ttw


    
def process_testing_captures_parallel_distribution_ROAD(testing_capture, ground_truth_dbc_path, attack_metadata, freq,
                                      corr_sample_training, method, dataset):
    
    print("computing: ", testing_capture)
    
    # dict to store computations
    resulting_dic = defaultdict(dict)

    # extract injection intervals
    injection_interval = attack_metadata[testing_capture[12:-14]]["injection_interval"]

    # Windowing datasets
    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            ground_truth, predict_proba, ttw = process_testing_capture_distribution_ROAD(testing_capture, ground_truth_dbc_path, freq, window, 
                                                        offset, corr_sample_training, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
            resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba
            resulting_dic[f"{window}-{offset}"]["ttw"] = ttw

    # Storing the file
    with open(f"/home/cloud/ceph-robust/CAN/signal-ids-benchmark/data/results_{testing_capture[12:-14]}_{method}_{dataset}.json", "w") as outfile:
        json.dump(resulting_dic, outfile)



def process_testing_capture_correlation_ROAD(testing_capture_name, ground_truth_dbc_path, freq, window, 
                                             offset, corr_matrices_training, signals_training, injection_interval):

    corr_matrices_testing, timepts = compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset)
    total_length = timepts[-1]
    # print("corr_matrices_testing: ", len(corr_matrices_testing))

    intervals_testing = create_time_intervals(total_length, window/freq, offset/freq)
    # print(intervals_testing[0:10])
    print("interval_testing: ", len(intervals_testing))

    ground_truth = []
    predict_proba = []

    start = time.time()

    for index_interval in range(len(intervals_testing)):
        
        # Compute signal names intersection
        signals_testing = corr_matrices_testing[index_interval].columns.values
        signal_names_intersection = list(set(signals_training).intersection(set(signals_testing)))

        # Filter correlation matrices by common names
        corr_matrix_1 = corr_matrices_training[0].loc[signal_names_intersection, signal_names_intersection]
        corr_matrix_2 = corr_matrices_testing[index_interval].loc[signal_names_intersection, signal_names_intersection]

        try:
        # Do hypothesis test
        # spearman_test = spearmanr(upper(corr_matrix_1), upper(corr_matrix_2))[1]
            spearman_test = spearmanr(upper(corr_matrix_1), upper(corr_matrix_2)).pvalue
            # print((i, corr_matrix_1.shape[0], spearman_test[0], spearman_test[1]))

        except:
            # print("Problem with Spearman")
            # print(upper(corr_matrix_1))
            # print(upper(corr_matrix_2))

            # Evaluation criteria
            if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                    or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
                ground_truth.append(1)
                predict_proba.append(0)
            else:
                ground_truth.append(0)
                predict_proba.append(0)

        else:

            # Evaluation criteria
            if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                    or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
                ground_truth.append(1)
                predict_proba.append(spearman_test)
            else:
                ground_truth.append(0)
                predict_proba.append(spearman_test)

    end = time.time()

    ttw = (end - start)/len(intervals_testing)

    return ground_truth, predict_proba, ttw



def process_testing_captures_parallel_correlation_ROAD(testing_capture, ground_truth_dbc_path, attack_metadata, freq,
                                      corr_matrices_training, signals_training, method, dataset):
    
    print("computing: ", testing_capture)
    
    # dict to store computations
    resulting_dic = defaultdict(dict)

    # extract injection intervals
    injection_interval = attack_metadata[testing_capture[12:-14]]["injection_interval"]

    # Windowing datasets
    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            ground_truth, predict_proba, ttw = process_testing_capture_correlation_ROAD(testing_capture, ground_truth_dbc_path, freq, window, 
                                                        offset, corr_matrices_training, signals_training, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
            resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba
            resulting_dic[f"{window}-{offset}"]["ttw"] = ttw

    # Storing the file
    with open(f"/home/cloud/ceph-robust/CAN/signal-ids-benchmark/data/results_{testing_capture[12:-14]}_{method}_{dataset}.json", "w") as outfile:
        json.dump(resulting_dic, outfile)



def compute_distance_matrix(corr_matrix):

    signal_names = np.array(corr_matrix.columns)

    # display(corr_matrix)

    # compute distance matrix
    # distance_matrix = np.sqrt(2*(1 - corr_matrix.to_numpy())) 
    distance_matrix = 2*(1 - corr_matrix.to_numpy())
    distance_matrix[distance_matrix < 0] = 0
    # display(distance_matrix.shape)
    # display(distance_matrix)

    return signal_names, distance_matrix



def process_testing_capture_DBSCAN_ROAD(testing_capture_name, ground_truth_dbc_path, freq, window, offset, injection_interval):
    
    # print("Hola")

    # DBSCAN Object
    DBSCAN_clustering = DBSCAN(eps=1, min_samples=1, metric="precomputed")

    corr_matrices_testing, timepts = compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset)
    total_length = timepts[-1]
    # print("corr_matrices_testing: ", len(corr_matrices_testing))

    intervals_testing = create_time_intervals(total_length, window/freq, offset/freq)
    # print(intervals_testing[0:10])
    print("interval_testing: ", len(intervals_testing))

    ground_truth = []
    predict_proba = [] 

    start = time.time()

    for index_interval in range(len(intervals_testing)):

        # print("Interval: ", intervals_testing[index_interval])

        # print(np.isnan(corr_matrices_testing[index_interval]).any().any())
        # print((corr_matrices_testing[index_interval] < 0).any().any())

        # print(corr_matrices_testing[index_interval].shape)#, corr_matrices_testing[index_interval])
        # print(corr_matrices_testing[index_interval].any())

        # Check if there are elements in the correlation matrix
        if corr_matrices_testing[index_interval].shape != (0, 0):

            signal_names, distance_matrix = compute_distance_matrix(corr_matrices_testing[index_interval])
            # print(np.isnan(distance_matrix).any().any())
            # print((distance_matrix < 0))
            # display(distance_matrix[distance_matrix < 0])
            # print(type(signal_names), signal_names)
            
            # print(distance_matrix)
            DBSCAN_clustering.fit(distance_matrix)

            clustering_labels = DBSCAN_clustering.labels_
            
            unique_clustering_labels = np.unique(clustering_labels)

            # print(len(clustering_labels), len(unique_clustering_labels), clustering_labels)

            max_error_all_clusters = []

            for cluster_id in (unique_clustering_labels):
                
                index_of_interest = np.argwhere(clustering_labels == cluster_id).flatten()
                # print(cluster_id, len(index_of_interest), index_of_interest)
                # print(signal_names[index_of_interest])
                
                if len(index_of_interest) >= 2: # Check only clusters with at least two elements

                    pd_corr_matrix = pd.DataFrame(corr_matrices_testing[index_interval], index=signal_names, columns=signal_names)
                    # display(pd_distance_matrix)
                    matrix_of_interest = pd_corr_matrix.loc[signal_names[index_of_interest], signal_names[index_of_interest]]
                    # display(matrix_of_interest)

                    upper_matrix_of_interest = upper(matrix_of_interest)
                    # display(upper_matrix_of_interest)

                    mean_cluster = np.mean(upper_matrix_of_interest)
                    std_cluster = np.std(upper_matrix_of_interest)
                    # print(std_cluster)

                    if std_cluster != 0:

                        error_cluster = np.absolute(upper_matrix_of_interest - mean_cluster)
                        error_cluster = error_cluster/std_cluster
                        max_error_cluster = np.max(error_cluster)
                        
                        max_error_all_clusters.append(max_error_cluster)

                    # break

            if len(max_error_all_clusters) != 0:
                mean_max_error_all_clusters = np.mean(max_error_all_clusters)
                std_max_error_all_clusters = np.std(max_error_all_clusters)
                detect_probability = norm.cdf(mean_max_error_all_clusters)

                # print("mean: ", mean_max_error_all_clusters, "std: ", std_max_error_all_clusters, "dist: ", max_error_all_clusters)
            else:
                detect_probability = 0 
            
            # break

            if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                    or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
                ground_truth.append(1)
                predict_proba.append(detect_probability)
            else:
                ground_truth.append(0)
                predict_proba.append(detect_probability)

        # Assign probability of detection 0 when there is an empty slice
        else:
            if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                    or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
                ground_truth.append(1)
                predict_proba.append(0)
            else:
                ground_truth.append(0)
                predict_proba.append(0)

    end = time.time()

    ttw = (end - start)/len(intervals_testing)

    return ground_truth, predict_proba, ttw




def process_testing_captures_parallel_DBSCAN_ROAD(testing_capture, ground_truth_dbc_path, attack_metadata, freq, method, dataset):

    print("computing: ", testing_capture)
    
    # dict to store computations
    resulting_dic = defaultdict(dict)

    # extract injection intervals
    injection_interval = attack_metadata[testing_capture[12:-14]]["injection_interval"]

    # Windowing datasets
    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            ground_truth, predict_proba, ttw = process_testing_capture_DBSCAN_ROAD(testing_capture, ground_truth_dbc_path, freq, window, offset, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
            resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba
            resulting_dic[f"{window}-{offset}"]["ttw"] = ttw

    # Storing the file
    with open(f"/home/cloud/ceph-robust/CAN/signal-ids-benchmark/data/results_{testing_capture[12:-14]}_{method}_{dataset}.json", "w") as outfile:
        json.dump(resulting_dic, outfile)


def hierarchical_clustering(corr_matrix, method="complete"):
    
    if method == "complete":
        Z = complete(corr_matrix)
    if method == "single":
        Z = single(corr_matrix)
    if method == "average":
        Z = average(corr_matrix)
    if method == "ward":
        Z = ward(corr_matrix)
  
    # PLotting the dendrogram
    # fig = plt.figure(figsize=(16, 8))
    # dn = dendrogram(Z)
    # plt.title(f"Dendrogram for {method}-linkage with correlation distance")
    # plt.show()
    
    return Z


def compute_hierarchical_clustering(corr_matrix_1, corr_matrix_2, signal_names_intersection, method):
    
    # Filter correlation matrices by common names
    corr_matrix_1 = corr_matrix_1.loc[signal_names_intersection, signal_names_intersection]
    # display(corr_matrix_1)

    corr_matrix_2 = corr_matrix_2.loc[signal_names_intersection, signal_names_intersection]
    # display(corr_matrix_2)
    
    linkage_matrix_1 = hierarchical_clustering(corr_matrix_1, method=method)
    linkage_matrix_2 = hierarchical_clustering(corr_matrix_2, method=method)
    
    return linkage_matrix_1, linkage_matrix_2



def compute_element_centric_similarity(linkage_matrix_1, linkage_matrix_2, r=1.0):
    
    c_1 = Clustering().from_scipy_linkage(linkage_matrix_1, dist_rescaled=True)
    c_2 = Clustering().from_scipy_linkage(linkage_matrix_2, dist_rescaled=True)
    
    return sim.element_sim(c_1, c_2, r=r, alpha=0.9)



def process_testing_capture_AHC_ROAD(testing_capture_name, ground_truth_dbc_path, freq, window, offset, corr_matrices_training, signals_training, injection_interval):

    corr_matrices_testing, timepts = compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset)
    total_length = timepts[-1]
    # print("corr_matrices_testing: ", len(corr_matrices_testing))

    intervals_testing = create_time_intervals(total_length, window/freq, offset/freq)
    # print(intervals_testing[0:10])
    print("interval_testing: ", len(intervals_testing))

    ground_truth = []
    predict_proba = [] 

    start = time.time()

    for index_interval in range(len(intervals_testing)):

        # print("Interval: ", intervals_testing[index_interval])

        # print(np.isnan(corr_matrices_testing[index_interval]).any().any())
        # print((corr_matrices_testing[index_interval] < 0).any().any())

        # print("Interval: ", intervals_testing[index_interval])

        # print(np.isnan(corr_matrices_testing[index_interval]).any().any())
        # print((corr_matrices_testing[index_interval] < 0).any().any())

        # Check if there are elements in the correlation matrix
        if corr_matrices_testing[index_interval].shape != (0, 0):

            signal_names_testing = corr_matrices_testing[index_interval].columns.values
            # print(type(signal_names_testing), signal_names_testing)

            signal_names_intersection = list(set(signals_training).intersection(set(signal_names_testing)))

            linkage_matrix_training, linkage_matrix_testing = compute_hierarchical_clustering(corr_matrices_training, corr_matrices_testing[index_interval], signal_names_intersection, "ward")

            similarity = 1 - compute_element_centric_similarity(linkage_matrix_training, linkage_matrix_testing, r=-5)
            # print("similarity: ", similarity)

            if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                        or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                            or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
                ground_truth.append(1)
                predict_proba.append(similarity)
            else:
                ground_truth.append(0)
                predict_proba.append(similarity)

        # Assign probability of detection 0 when there is an empty slice
        else:
            if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                    or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
                ground_truth.append(1)
                predict_proba.append(0)
            else:
                ground_truth.append(0)
                predict_proba.append(0)

    end = time.time()

    ttw = (end - start)/len(intervals_testing)

    return ground_truth, predict_proba, ttw



def process_testing_captures_parallel_AHC_ROAD(testing_capture, ground_truth_dbc_path, attack_metadata, freq, method, dataset, corr_matrices_training, signals_training):

    print("computing: ", testing_capture)
    
    # dict to store computations
    resulting_dic = defaultdict(dict)

    # extract injection intervals
    injection_interval = attack_metadata[testing_capture[12:-14]]["injection_interval"]

    # Windowing datasets
    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            ground_truth, predict_proba, ttw = process_testing_capture_AHC_ROAD(testing_capture, ground_truth_dbc_path, freq, window, offset, 
                                                                           corr_matrices_training, signals_training, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
            resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba
            resulting_dic[f"{window}-{offset}"]["ttw"] = ttw

    # Storing the file
    with open(f"/home/cloud/ceph-robust/CAN/signal-ids-benchmark/data/results_{testing_capture[12:-14]}_{method}_{dataset}.json", "w") as outfile:
        json.dump(resulting_dic, outfile)


def count_positive_windows(testing_capture_name, ground_truth_dbc_path, freq, window, offset, injection_interval):

    _, timepts = compute_correlation_distribution_testing(testing_capture_name, ground_truth_dbc_path, freq, window, offset)
    total_length = timepts[-1]
    # print("corr_matrices_testing: ", len(corr_matrices_testing))

    intervals_testing = create_time_intervals(total_length, window/freq, offset/freq)
    # print(intervals_testing[0:10])
    # print("interval_testing: ", len(intervals_testing))

    counter = 0
    positive_windows = 0

    for index_interval in range(len(intervals_testing)):

        # print("Interval: ", intervals_testing[index_interval])

        counter += 1

        if ((intervals_testing[index_interval][1] > injection_interval[0] and intervals_testing[index_interval][0] < injection_interval[0])
                    or (intervals_testing[index_interval][0] > injection_interval[0] and intervals_testing[index_interval][1] < injection_interval[1])
                        or (intervals_testing[index_interval][0] < injection_interval[1] and intervals_testing[index_interval][1] > injection_interval[1])):
            positive_windows += 1

    return positive_windows/counter                                                                   


def count_positive_windows_parallel(testing_capture, ground_truth_dbc_path, freq, attack_metadata, dataset):

    print("computing: ", testing_capture)
    
    # dict to store computations
    resulting_dic = defaultdict(dict)

    # extract injection intervals
    injection_interval = attack_metadata[testing_capture[12:-14]]["injection_interval"]

    # Windowing datasets
    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            positive_proportion = count_positive_windows(testing_capture, ground_truth_dbc_path, freq, window, offset, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["positive_proportion"] = positive_proportion

    # Storing the file
    with open(f"/home/cloud/Projects/CAN/signal-ids-benchmark/data/results_positive_proportion_{testing_capture[12:-14]}_{dataset}.json", "w") as outfile:
        json.dump(resulting_dic, outfile)