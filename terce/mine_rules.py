#Author: Omar Bahri

import numpy as np
import os
import sys
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import LabelEncoder
from ruletransform import ContractedRuleTransform
from ruletransform.data_io import load_from_tsfile_to_dataframe
from utils import save_transformer

def main():
    name = sys.argv[1]

    X_train, y_train = load_from_tsfile_to_dataframe(
        os.path.abspath(os.path.join('..', 'data', name, name + "_TRAIN.ts"))
    )
    X_test, y_test = load_from_tsfile_to_dataframe(
            os.path.abspath(os.path.join('..', 'data', name, name + "_TEST.ts"))
        )
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # How long (in minutes) to extract shapelets for.
    # This is a simple lower-bound initially;
    # once time is up, no further shapelets will be assessed
    time_contract_in_mins = int(sys.argv[2])
    
    #Set lengths of shapelets to mine
    max_perc = float(sys.argv[3])
    min_length, max_length = 5, int(max_perc/100*X_train.shape[1])
    
    #Mine rules using RT
    rt = ContractedRuleTransform(
        shapelet_mining_contract=time_contract_in_mins,
        rule_mining_contract=5,         #5 minutes, just enough to get enough rules for TeRCE
        min_shapelet_length=min_length,
        max_shapelet_length=max_length,
        occ_threshold = 10,
        verbose=0,
    )
    
    #Given the short mining contract, this example only uses the first two dimensions of the dataset
    rt.fit(X_train, y_train)
    all_rules_counts, all_rules_indices, all_rules_shs = rt.transform(X_train, test=False)
    
    all_rules_counts[all_rules_counts>0] = 1
    
    #Get fisher scores and sort list of indices
    scores = fisher_score.fisher_score(all_rules_counts, y_train)
    best_rules_indices = np.argsort(scores)[::-1]
    
    del scores
            
    #name of current run (dataset + parameters combination)
    run_name = '_'.join(name, str(time_contract_in_mins), str(max_perc))
    
    #path of intermediary results directory
    inter_results = os.path.abspath(os.path.join('results', 'util_data', run_name))
    
    if not os.path.exists(inter_results):
        os.makedirs(inter_results)
    
    np.save(os.path.join(inter_results, "all_rules_counts.npy"),
            all_rules_counts[:,best_rules_indices])
    np.save(os.path.join(inter_results, "all_rules_indices.npy"),
            all_rules_indices[:,best_rules_indices])
    np.save(os.path.join(inter_results, "all_rules_shs.npy"),
            all_rules_shs[:,best_rules_indices])        
    
    del all_rules_counts, all_rules_indices, all_rules_shs
    
    #save shapelets from ST
    save_transformer(inter_results, rt._transformer)
    
    #mine rules from test set
    _, all_rules_indices_test = rt.transform(X_test, test=True)
    
    np.save(os.path.join(inter_results, "all_rules_indices_test.npy"),
            all_rules_indices_test[:,best_rules_indices])
    
if __name__ == "__main__":
    main()