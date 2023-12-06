#Author: Omar Bahri

from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import random
import pickle
import sys
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime_convert import from_nested_to_3d_numpy

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def main():
    random.seed(42)

    name = sys.argv[1]
    time_contract_in_mins = int(sys.argv[2])
    max_perc = float(sys.argv[3])
    
    #name of current run (dataset + parameters combination)
    run_name = '_'.join(name, str(time_contract_in_mins), str(max_perc))
    
    #path of intermediary results directory
    inter_results = os.path.abspath(os.path.join('results', 'util_data', run_name))
    
    #number of rules to keep per class, 50 should be plenty but feel free to experiment
    n_rc = 50
    
    #Load our training set
    X_train, y_train = load_from_tsfile_to_dataframe(
        os.path.abspath(os.path.join('..', 'data', name, name + "_TRAIN.ts"))
    )
    X_test, y_test = load_from_tsfile_to_dataframe(
            os.path.abspath(os.path.join('..', 'data', name, name + "_TEST.ts"))
        )
    
    X_train = from_nested_to_3d_numpy(X_train)
    y_train = np.asarray(y_train)
    X_test = from_nested_to_3d_numpy(X_test)
    y_test = np.asarray(y_test)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    rules_counts = np.load(os.path.join(inter_results, "all_rules_counts.npy"))
    rules_indices = np.load(os.path.join(inter_results, "all_rules_indices.npy"))
    
    ts_length = X_train.shape[2]
    
    #Get rules class occurences
    rules_classes = []
    for rule_idx in range(rules_counts.shape[1]):
        rule_classes = []
        for instance_idx, instance_counts in enumerate(rules_counts):
            if instance_counts[rule_idx] > 0:
                rule_classes.append(y_train[instance_idx])
        rules_classes.append(rule_classes)
    
    #initialize a dictionary that stores lists of class-rules
    rules_class = {}
    #initialize a dictionary that stores lists of class-shapelets heatmaps
    heat_maps = {}
    
    for c in np.unique(y_train):
        rules_class[c] = []
        heat_maps[c] = []
    
    for i, rule_classes in enumerate(rules_classes):
        if np.unique(rule_classes).shape[0] == 1:
            # print('rules_class ', i, np.unique(rule_classes))
            for c in np.unique(y_train):
                if np.unique(rule_classes)[0] == c:
                    rules_class[c].append(i)
    
    for c in np.unique(y_train):
        if len(rules_class[c]) >= n_rc:
            rules_class[c] = rules_class[c][:n_rc]
    
    # Get shapelet_locations distributions per exclusive class
    for c in np.unique(y_train):
        for r_idx in rules_class[c]:
            heat_map_1 = np.zeros(ts_length)
            heat_map_2 = np.zeros(ts_length)
        
            num_occurences = 0
        
            for instance_idx in range(rules_indices.shape[0]):
                if len(rules_indices[instance_idx][r_idx]) == 0:
                    continue
                else:
                    for (s1, e1, s2, e2) in rules_indices[instance_idx][r_idx]:
                        for idx in range(s1, e1):
                            heat_map_1[idx] += 1
                        for idx in range(s2, e2):
                            heat_map_2[idx] += 1
                        num_occurences += 1
        
            heat_map_1 = heat_map_1/num_occurences
            heat_map_2 = heat_map_2/num_occurences
        
            heat_maps[c][r_idx] = (heat_map_1, heat_map_2)
          
    #save intermediate results
    np.save(os.path.join(inter_results, 'heat_maps.npy'), heat_maps)
    np.save(os.path.join(inter_results, 'rules_class.npy'), rules_class)

if __name__ == "__main__":
    main()