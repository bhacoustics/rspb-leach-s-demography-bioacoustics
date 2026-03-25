import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch


def known_unknown_split(df, unknown_burrows, bad_ind):
    df[['location_id', 'sex']] = df['individual_id'].str.split('-', expand=True)
    mask = df['location_id'].isin(unknown_burrows)

    df_unknown = df[mask]
    df_known = df[~mask]

    df_unknown = df_unknown[~((df_unknown['collection_id'] == 'SK-2024') & (df_unknown['individual_id'].isin(bad_ind)))] 
    df_unknown.reset_index(drop=True, inplace=True)

    return df_known, df_unknown

def get_audio_id(df, collection, prefix, pattern):
    extracted = df['individual_id'].str.extract(pattern)
    year_suffix = collection[-2:]
    df['audio_id'] = prefix + extracted[0].fillna('') + extracted[1].fillna('') + year_suffix

    return df

# creates train df for first year of data

def get_train_df(df_known, train_df, collection, DICT, MIN_CALLS, prefix, pattern):
    
    print(f"Proccessing {collection}")

    df = df_known[df_known['collection_id'] == collection].copy()
    v = df[df['augmentation'] == 0].individual_id.value_counts()
    df = df[df.individual_id.isin(v.index[v.gt(MIN_CALLS)])]
    df = get_audio_id(df, collection, prefix, pattern)
    id_df = df[['individual_id', 'audio_id']].copy()
    id_map = pd.Series(id_df.audio_id.values, index=id_df.individual_id).drop_duplicates().to_dict()
    DICT[collection] = id_map
    DICT['id_map_all'] = id_map

    train_df = pd.concat([train_df, df])
    num_individuals = train_df['audio_id'].nunique()

    print(f"Training df includes {train_df['label_id'].nunique()} calls from {num_individuals} individuals in {collection}\n")
    print(f"Mapped audio_id and individual_id for {collection}")

    return train_df

# Adds to train df using new predictions/labels from subsequent years

def append_train_df(df_known, train_df, collection, DICT, MIN_CALLS):

    df = df_known[df_known['collection_id'] == collection].copy()
    v = df[df['augmentation'] == 0].individual_id.value_counts()
    df = df[df.individual_id.isin(v.index[v.gt(MIN_CALLS)])]
    coll_map = DICT[collection]
    df['audio_id'] = df['individual_id'].map(coll_map)

    train_df = pd.concat([train_df, df])

    return train_df

# this is for setting a min and max number of calls per individual
# drops individuals with less than this number of calls 

def undersample_calls(df, MAX_CALLS, RANDOM_SEED):
   
    # mapping labels to individuals
    unaugmented_df = df[df['augmentation'] == 0].copy()
    id_groups = unaugmented_df.groupby('audio_id')['label_id'].unique()
    
    kept_label_ids = []
    # Filtering
    for ind_id, label_list in id_groups.items():
        n = len(label_list)
        
        if n > MAX_CALLS:
            np.random.seed(RANDOM_SEED)
            selected = np.random.choice(label_list, size=MAX_CALLS, replace=False)
            kept_label_ids.extend(selected)
        
        else:
            kept_label_ids.extend(label_list)
            
    # df now contains only filtered individuals/calls
    filter_df = df[df['label_id'].isin(kept_label_ids)].reset_index(drop=True)
    individual_classes = filter_df['audio_id'].nunique()
    
    # what's left ...
    print(f"After processing: Training df includes {filter_df['label_id'].nunique()} calls from {individual_classes} individuals.")

    return filter_df

# This splits the test and validation sets, ensuring a good representation of classes

def train_val_split(df, df_unknown, VAL_SIZE, UNK_RATIO):
    # Create split in unaugmented samples, introducing the rest afterwards to avoid leakage

    le = LabelEncoder()
    df['idx_label'] = le.fit_transform(df['audio_id'])
    df_unknown['idx_label'] = -1
    
    known_df = df[df['augmentation'] == 0]
    unknown_df = df_unknown[df_unknown['augmentation'] == 0]
    
    # Split 'knowns' according to VAL_SIZE with good representation
    train_known_ids, val_known_ids = train_test_split(known_df['label_id'].values, test_size=VAL_SIZE, random_state=42, stratify=known_df['audio_id'].values)
    train_unk_ids, val_unk_ids = train_test_split(unknown_df['label_id'].values, test_size=VAL_SIZE, random_state=42)
    
    # Make val_df using correct ratio of 'known' to 'unkown'
    val_known = known_df[known_df['label_id'].isin(val_known_ids)]
    n_unk_val = int((UNK_RATIO * len(val_known)) / (1 - UNK_RATIO))
    val_unk = unknown_df[unknown_df['label_id'].isin(val_unk_ids)].sample(n=min(n_unk_val, len(val_unk_ids)), random_state=42)
    val_df = pd.concat([val_known, val_unk]).sample(frac=1)
    
    # Make train_df using correct ratio of 'known' to 'unkown'
    train_known = df[df['label_id'].isin(train_known_ids)]
    
    n_unk_train = int((UNK_RATIO * len(train_known)) / (1 - UNK_RATIO))
    # checks max number of 'unkown' calls available
    num_all_unknown = df_unknown[df_unknown['label_id'].isin(train_unk_ids)]
    # samples using the ratio and max available, and shuffles the train_df
    train_unk = df_unknown[df_unknown['label_id'].isin(train_unk_ids)].sample(n=min(n_unk_train, len(num_all_unknown)), random_state=42)
    train_df = pd.concat([train_known, train_unk]).sample(frac=1)
    
    unk_pct_val = (val_df['idx_label'] == -1).sum() / len(val_df) * 100
    unk_pct_train = (train_df['idx_label'] == -1).sum() / len(train_df) * 100
    
    print(f"Validation size: {len(val_df)} (Unknown: {unk_pct_val:.2f}%)")
    print(f"Training size (inc. aug): {len(train_df)} (Unknown: {unk_pct_train:.2f}%)\n")
    
    return train_df, val_df

# Calculating class weights to be used in cross entropy loss

def get_class_weights(df, train_df, device):
    
    all_labels = df['audio_id'].unique()
    known_labels = sorted(all_labels)
    known_train_labels = train_df[train_df['idx_label'] != -1]['idx_label'].values
        
    raw_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(known_labels)), y=known_train_labels)
    class_weights = torch.tensor(raw_weights, dtype=torch.float).to(device)

    return class_weights, known_labels


