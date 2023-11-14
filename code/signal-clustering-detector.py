from helpers import *
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
from multiprocessing import Process
import sys

def main():

    # Get arguments on the command line
    method = sys.argv[1]
    dataset = sys.argv[2]

    # print(method) 
    # print(dataset)

    training_captures, testing_captures = get_captures_names()
    # print(len(testing_captures), testing_captures, "\n")
    
    attack_metadata, attack_metadata_keys = get_metadata_info()
    # print(len(attack_metadata_keys), attack_metadata_keys, "\n")

    ground_truth_dbc_path = get_dbc_path()

    corr_matrices_training, signals_training = compute_correlation_training(training_captures[-1], ground_truth_dbc_path, freq=100)
    # print(len(corr_matrices_training), type(corr_matrices_training[0]))
  
    # Execute detection
    window = 450
    offset = 10

    resulting_dic = defaultdict(dict)
    testing_capture = testing_captures[4] # Select the testing capture
    print(testing_capture)
    injection_interval = attack_metadata[testing_capture[12:-14]]["injection_interval"]
    freq = 100

    # Trying our different conbinations of window length and offset
    for window in tqdm(np.arange(50, 450, 50)):
        for offset in np.arange(10, window + 10, 10):

            print(f"window:{window}, offset:{offset}")

            ground_truth, predict_proba = process_testing_capture_AHC_ROAD(testing_capture, ground_truth_dbc_path, freq, window, offset, 
                                                                           corr_matrices_training[0], signals_training, injection_interval)
    
            resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
            resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba

            # break

        # break

    # print(ground_truth) 
    # print(predict_proba)

    print(dict(resulting_dic))

if __name__ == "__main__": 
    main() 