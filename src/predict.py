from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

# loading next collection to predict

def create_test_df(df, test_coll, MIN_CALLS):
    
    test_df = df[df['collection_id'] == test_coll]
    test_df = test_df[test_df['augmentation'] == 0]
    
    # Only test on individuals with with at least 5 calls
    v = test_df.individual_id.value_counts()
    test_df = test_df[test_df.individual_id.isin(v.index[v.gt(MIN_CALLS)])]
    test_df.reset_index(drop=True, inplace=True)
    test_num_individuals = test_df['individual_id'].nunique()
    
    print(f"\nTest df contains {test_df['label_id'].nunique()} calls from {test_num_individuals} individuals in {test_coll}\n")

    return test_df

def group_probabilities(model, test_df, known_labels):
    model.eval()
    embeddings = torch.tensor(np.stack(test_df['embedding'].values), dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(embeddings) 

    print("Train df processed by model")
        
    logit_df = pd.DataFrame(logits.numpy(), columns=known_labels)
    logit_df['individual_id'] = test_df['individual_id'].values

    # averaging for individual
    group_logits = logit_df.groupby('individual_id').mean()
    
    # Softmax of grouped logits
    group_probs = torch.softmax(torch.tensor(group_logits.values), dim=1).numpy()
    
    result_df = pd.DataFrame(group_probs, columns=known_labels, index=group_logits.index)
    
    # Calculate entropy to indicate 'unknown' score
    group_probs_torch = torch.tensor(group_probs)
    entropy = -torch.sum(group_probs_torch * torch.log(group_probs_torch + 1e-10), dim=1)
    result_df['unknown'] = (entropy / np.log(len(known_labels))).numpy()
    
    print("Probabilities grouped by individual")
    
    return result_df

