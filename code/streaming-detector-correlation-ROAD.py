from helpers import *
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
from multiprocessing import Process
import sys

def main():

    method = sys.argv[1] # distribution, correlation, DBSCAN, AHC
    dataset = sys.argv[2] # ROAD, SYNCAN

    # print(method) 
    # print(dataset)
    # print(os.getcwd())

    # print("Hola")
    training_captures, testing_captures = get_captures_names()
    # print(len(testing_captures), testing_captures, "\n")
    
    attack_metadata, attack_metadata_keys = get_metadata_info()
    # print(len(attack_metadata_keys), attack_metadata_keys, "\n")

    ground_truth_dbc_path = get_dbc_path()

    # signal_multivar_ts, timepts, aid_signal_tups = from_capture_to_time_series(training_captures[-1], ground_truth_dbc_path, freq=100)
    # print(signal_multivar_ts.shape)
    # print(np.diff(timepts))

    corr_matrices_training, signals_training = compute_correlation_training(training_captures[-1], ground_truth_dbc_path, freq=100)
    # print(testing_captures[2])
    # print(testing_captures[0][12:-14])
    
    # print(injection_interval)
    
    #print(len(corr_sample_training))
    #display(corr_sample_training)

    ############################################

    # # window = 100
    # # offset = 10

    resulting_dic = defaultdict(dict)
    injection_interval = attack_metadata[testing_captures[7][12:-14]]["injection_interval"]
    freq = 100

    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            print(f"window:{window}, offset:{offset}")

            ground_truth, predict_proba = process_testing_capture_correlation_ROAD(testing_captures[7], ground_truth_dbc_path, freq, window, 
                                                        offset, corr_matrices_training, signals_training, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
            resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba

    print(ground_truth) 
    print(predict_proba)

    print(dict(resulting_dic))

    # with open(f"/home/cloud/Projects/CAN/signal-ids-benchmark/data/results_{testing_captures[0][12:-14]}_{method}_{dataset}.json", "w") as outfile:
    #     json.dump(resulting_dic, outfile)

    ############################################

    # jobs = []
    # freq = 100

    # for testing_capture in testing_captures:

    #     p = Process(target=process_testing_captures_parallel_correlation_ROAD, args=(testing_capture, ground_truth_dbc_path, 
    #                                                                                  attack_metadata, freq, corr_matrices_training, signals_training, 
    #                                                                                  method, dataset))
        
    #     jobs.append(p)
    #     p.start()

    # # Wait for this [thread/process] to complete
    # for proc in jobs:
    #     proc.join()

    

if __name__ == "__main__": 
    main() 