from helpers import *
import numpy as np
from collections import defaultdict

def main():

    # print("Hola")
    training_captures, testing_captures = get_captures_names()
    # print(len(testing_captures), testing_captures, "\n")
    
    attack_metadata, attack_metadata_keys = get_metadata_info()
    # print(len(attack_metadata_keys), attack_metadata_keys, "\n")

    ground_truth_dbc_path = get_dbc_path()

    # signal_multivar_ts, timepts, aid_signal_tups = from_capture_to_time_series(training_captures[-1], ground_truth_dbc_path, freq=100)
    # print(signal_multivar_ts.shape)
    # print(np.diff(timepts))

    corr_sample_training, signals_training = compute_correlation_distribution_training(training_captures[-1], ground_truth_dbc_path, freq=100)
    # print(testing_captures[0])
    # print(testing_captures[0][12:-14])
    injection_interval = attack_metadata[testing_captures[0][12:-14]]["injection_interval"]
    # print(injection_interval)
    
    #print(len(corr_sample_training))
    #display(corr_sample_training)

    ############################################

    window = 50
    offset = 1


    for window in np.arange(50, 550, 50):
        for offset in np.arange(10, window + 10, 10):
            print(window, offset)

    # ground_truth, predict_proba = process_testing_capture_correlation_ROAD(testing_captures[0], ground_truth_dbc_path, freq=100, window=window, 
    #                                                     offset=offset, corr_sample_training=corr_sample_training, injection_interval=injection_interval)
    
    # resulting_dic = defaultdict(dict)
    
    # resulting_dic[f"{window}-{offset}"]["ground_truth"] = ground_truth
    # resulting_dic[f"{window}-{offset}"]["predict_proba"] = predict_proba

    # print(ground_truth) 
    # print(predict_proba)

    # print(dict(resulting_dic))

    

if __name__ == "__main__": 
    main() 