#Author: Omar Bahri

from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import sys
import random
import itertools
from tslearn.neighbors import KNeighborsTimeSeries
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime_convert import from_nested_to_3d_numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket

# Get the locations where the rules in rules_class occur in idx
def get_rules_locations(idx, rules_indices, rules_class):
    all_locs = {}

    for rule_idx in rules_class:
        if rules_indices[idx][rule_idx] != -1:
            if len(rules_indices[idx][rule_idx]) > 0:
                all_locs[rule_idx] = rules_indices[idx][rule_idx]

    return all_locs

# Optimize by fitting outside or returning a list of all nns at once
def get_nearest_neighbor(knn, X_test, y_test, y_train, idx):
    # pred_label = y_pred[idx]-
    pred_label = y_test[idx]
    target_labels = np.argwhere(y_train != pred_label)

    X_test_knn = X_test[idx].reshape(1, X_test.shape[1], X_test.shape[2])
    X_test_knn = np.swapaxes(X_test_knn, 1, 2)

    _, nn = knn.kneighbors(X_test_knn)
    nn_idx = target_labels[nn][0][0][0]

    return nn_idx


def get_real_dim(dim, bad_dims):
    return dim - np.where(np.asarray(bad_dims) < dim)[0].shape[0]

def main():
    random.seed(42)
    np.random.seed(42)
    
    name = sys.argv[1]
    time_contract_in_mins = int(sys.argv[2])
    max_perc = float(sys.argv[3])
    
    #name of current run (dataset + parameters combination)
    run_name = '_'.join(name, str(time_contract_in_mins), str(max_perc))
    
    #path of intermediary results directory
    inter_results = os.path.abspath(os.path.join('results', 'util_data', run_name))
    
    #path of CFs results directory
    results = os.path.abspath(os.path.join('results', 'cfs', run_name))
    if not os.path.exists(results):
        os.makedirs(results)
    
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
    
    st_shapelets = np.load(os.path.join(inter_results, 'shapelets.pkl'), allow_pickle=True)    
    rules_class = np.load(os.path.join(inter_results, 'rules_class.npy'), allow_pickle=True).item()
    heat_maps = np.load(os.path.join(inter_results, 'heat_maps.npy'), allow_pickle=True).item()
    rules_indices = np.load(os.path.join(inter_results, 'all_rules_indices_test.npy'), allow_pickle=True)
    rules_shapelets = np.load(os.path.join(inter_results, 'all_rules_shs.npy'), allow_pickle=True)
    
    ts_length = X_train.shape[2]
    
    #Train the black-box model (ROCKET here, feel free to use different models)
    model = make_pipeline(
        Rocket(num_kernels=500), RandomForestClassifier(n_estimators=500)
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    y_pred = predictions
    
    #dictionary to store class KNNs
    knns = {}
    
    #Fit a KNN for each class
    for c in np.unique(y_train):
        knns[c] = KNeighborsTimeSeries(n_neighbors=1)
        X_train_knn = X_train[np.argwhere(y_train==c)].reshape(np.argwhere(y_train==c).shape[0],
                                 X_train.shape[1], X_train.shape[2])
        X_train_knn = np.swapaxes(X_train_knn, 1, 2)
        knns[c].fit(X_train_knn)
    
    #Generate CFs for each time series in the test set, for all possible target classes
    for instance_idx in range(0, X_test.shape[0]):
        orig_c = y_pred[instance_idx]
    
        # print("instance_idx: " + str(instance_idx))
        # print("from: " + str(orig_c))
    
        for target_c in set(np.unique(y_train)) - set([orig_c]):
            # print("to: ", + str(target_c))
            original_rules_class = rules_class[c]
            target_knn = knns[c]
            target_rules_class = rules_class[target_c]
            target_heat_maps = heat_maps[target_c]
          
            #get NUN
            nn_idx = get_nearest_neighbor(target_knn, X_test, y_test, y_train, instance_idx)
            
            all_locs = get_rules_locations(instance_idx, rules_indices, original_rules_class)
        
            cf = X_test[instance_idx].copy()
        
            # Remove original class rules
            for rule_idx in all_locs:
                cf_pred = model.predict(np.array([cf]))
        
                if cf_pred != target_c:
                    # print('Removing original rule: ', rule_idx)
        
                    dim1, _, dim2, _ = rules_shapelets[:, rule_idx]
        
                    for (s1, e1, s2, e2) in all_locs.get(rule_idx):
                        cf_pred = model.predict(np.array([cf]))
        
                        if cf_pred != target_c:
                            # print('Removing original rule occurence (antecedent)')
        
                            target_shapelet = X_train[nn_idx][dim1][s1:e1]
        
                            s_min = target_shapelet.min()
                            s_max = target_shapelet.max()
                            t_min = X_test[instance_idx][dim1][s1:e1].min()
                            t_max = X_test[instance_idx][dim1][s1:e1].max()
        
                            if s_max-s_min == 0:
                                target_shapelet = (t_max+t_min)/2 * \
                                    np.ones(len(target_shapelet))
                            else:
                                target_shapelet = (t_max-t_min)*(target_shapelet-s_min)/\
                                                    (s_max-s_min)+t_min
                                                    
                            cf[dim1][s1:e1] = target_shapelet
                        else:
                            break
        
                        cf_pred = model.predict(np.array([cf]))
        
                        if cf_pred != target_c:
                            # print('Removing original rule occurence (subsequent)')
        
                            target_shapelet = X_train[nn_idx][dim2][s2:e2]
        
                            s_min = target_shapelet.min()
                            s_max = target_shapelet.max()
                            t_min = X_test[instance_idx][dim2][s2:e2].min()
                            t_max = X_test[instance_idx][dim2][s2:e2].max()
        
                            if s_max-s_min == 0:
                                target_shapelet = (t_max+t_min)/2 * \
                                    np.ones(len(target_shapelet))
                            else:
                                target_shapelet = (t_max-t_min)*(target_shapelet-s_min)/\
                                    (s_max-s_min)+t_min
        
                            cf[dim2][s2:e2] = target_shapelet
                        else:
                            break
        
            cf_pred = model.predict(np.array([cf]))
            
            if cf_pred != target_c:
        
                ncf = cf.copy()
                all_perturbations = []
        
                # Introduce new rules from the target class
                for target_rule_idx in target_rules_class:
                    cf = ncf.copy()
                    cf_pred = model.predict(np.array([cf]))
        
                    if cf_pred != target_c:
                        # print('Introducing target rule: ', int(target_rule_idx))
        
                        hm_1, hm_2 = target_heat_maps.get(int(target_rule_idx))
        
                        center_1 = (np.argwhere(
                            hm_1 > 0)[-1][0] - np.argwhere(hm_1 > 0)[0][0])//2 + np.argwhere(hm_1 > 0)[0][0]
                        center_2 = (np.argwhere(
                            hm_2 > 0)[-1][0] - np.argwhere(hm_2 > 0)[0][0])//2 + np.argwhere(hm_2 > 0)[0][0]
            
                        dim1, sh1, dim2, sh2 = rules_shapelets[:, int(target_rule_idx)]
            
                        target_shapelet_1 = st_shapelets[dim1][sh1][0]
                        target_shapelet_2 = st_shapelets[dim2][sh2][0]
        
                        target_shapelet_length_1 = target_shapelet_1.shape[0]
                        target_shapelet_length_2 = target_shapelet_2.shape[0]
        
                        start_1 = center_1 - target_shapelet_length_1//2
                        end_1 = center_1 + (target_shapelet_length_1 -
                                            target_shapelet_length_1//2)
                        if start_1 < 0:
                            end_1 = end_1 - start_1
                            start_1 = 0
                        if end_1 > ts_length:
                            start_1 = start_1 - (end_1 - ts_length + 1)
                            end_1 = ts_length - 1
        
                        start_2 = center_2 - target_shapelet_length_2//2
                        end_2 = center_2 + (target_shapelet_length_2 -
                                            target_shapelet_length_2//2)
                        if start_2 < 0:
                            end_2 = end_2 - start_2
                            start_2 = 0
                        if end_2 > ts_length:
                            start_2 = start_2 - (end_2 - ts_length + 1)
                            end_2 = ts_length - 1
        
                        s_min_1 = target_shapelet_1.min()
                        s_max_1 = target_shapelet_1.max()
                        t_min_1 = X_test[instance_idx][dim1][start_1:end_1].min()
                        t_max_1 = X_test[instance_idx][dim1][start_1:end_1].max()
        
                        s_min_2 = target_shapelet_2.min()
                        s_max_2 = target_shapelet_2.max()
                        t_min_2 = X_test[instance_idx][dim2][start_2:end_2].min()
                        t_max_2 = X_test[instance_idx][dim2][start_2:end_2].max()
        
                        if s_max_1-s_min_1 == 0:
                            target_shapelet_1 = (t_max_1+t_min_1) / \
                                2*np.ones(len(target_shapelet_1))
                        else:
                            target_shapelet_1 = (
                                t_max_1-t_min_1)*(target_shapelet_1-s_min_1)/(s_max_1-s_min_1)+t_min_1
        
                        if s_max_2-s_min_2 == 0:
                            target_shapelet_2 = (t_max_2+t_min_2) / \
                                2*np.ones(len(target_shapelet_2))
                        else:
                            target_shapelet_2 = (
                                t_max_2-t_min_2)*(target_shapelet_2-s_min_2)/(s_max_2-s_min_2)+t_min_2
        
                        cf[dim1][start_1:end_1] = target_shapelet_1
                        cf[dim2][start_2:end_2] = target_shapelet_2
        
                        all_perturbations.append([(dim1, start_1, end_1, target_shapelet_1),
                                                  (dim2, start_2, end_2, target_shapelet_2)])
        
                    cf_pred = model.predict(np.array([cf]))
                    
                if cf_pred != target_c:
                    # print("Trying combinations", len(all_perturbations))
                    for L in range(0, len(all_perturbations)+1):
                        for subset in itertools.combinations(all_perturbations, L):
                            if cf_pred != target_c:
                                if len(subset) >= 2:
                                    cf = ncf.copy()
                                    for perts in subset:
                                        (dim1, start_1, end_1,
                                         target_shapelet_1) = perts[0]
                                        (dim2, start_2, end_2,
                                         target_shapelet_2) = perts[1]
    
                                        cf[dim1][start_1:end_1] = target_shapelet_1
                                        cf[dim2][start_2:end_2] = target_shapelet_2
    
                                    cf_pred = model.predict(np.array([cf]))
        
            if cf_pred == target_c:
                # print('CF found!')
                np.save(os.path.join(results, str(instance_idx) + '_to_' + str(target_c) + '.npy'), cf)     
            
if __name__ == "__main__":
    main()