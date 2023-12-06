#Author: Omar Bahri

import numpy as np
import pickle
import os
        
#write transformer to file
def save_transformer(parent_dir, transformer):
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    with open(os.path.join(parent_dir, "shapelets.pkl"), 'wb') as f:
            pickle.dump(get_shapelets(transformer), f)
    np.save(os.path.join(parent_dir, "indices.npy"), get_indices(transformer))
    np.save(os.path.join(parent_dir, "scores.npy"), get_scores(transformer))
        
#save shapelets distances only (for test set)
def save_shapelets_distances(parent_dir, transformer, test=False):
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    if test:
        with open(os.path.join(parent_dir, "shapelets_distances_test.pkl"), 'wb') as f:
            pickle.dump(get_shapelets_distances(transformer), f)
    else:
        with open(os.path.join(parent_dir, "shapelets_distances.pkl"), 'wb') as f:
            pickle.dump(get_shapelets_distances(transformer), f)
    
#get the list of shapelets of a transformer
def get_shapelets(transformer):    
    all_shapelets = []
    
    for st in transformer.sts:
        dim_shapelets = []
        for shapelet in st.shapelets:
            dim_shapelets.append(shapelet.data)
        all_shapelets.append(dim_shapelets)
        
    return all_shapelets

#get the list of shapelet indices of a transformer
def get_indices(transformer):    
    all_indices = []
    
    for st in transformer.sts:
        dim_indices = []
        for shapelet in st.shapelets:
            ind = np.empty(3, dtype=np.uint16)
            ind[0] = shapelet.series_id
            ind[1] = shapelet.start_pos
            ind[2] = shapelet.start_pos + shapelet.length
            dim_indices.append(ind)
        all_indices.append(dim_indices)
        
    return np.asarray(np.asarray(all_indices))
    
#get the list of shapelet scores of a transformer
def get_scores(transformer):
    all_scores = []
    
    for st in transformer.sts:
        dim_scores = []
        for shapelet in st.shapelets:
            dim_scores.append(shapelet.info_gain)
        all_scores.append(dim_scores)
        
    return np.asarray(np.asarray(all_scores))
    
#get the distance of shapelets from each other shapelet in the MTS
def get_shapelets_distances(transformer):
    all_shapelets_distances = []
    
    for st in transformer.sts:
        shapelets_distances = []
        for shapelet in st.shapelets:
            shapelets_distances.append(shapelet.distances)
        
        all_shapelets_distances.append(shapelets_distances)
    return all_shapelets_distances