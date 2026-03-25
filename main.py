import os
import sys
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from src import *

def main():
    # create folder with timestamp to save results to
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    folder_name = Path(f"AIID_results_{timestamp}")
    folder_name.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved in: {folder_name}")

    # load config file
    with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

    # save a copy of the config file used
    config_source = Path("config.yaml")
    config_destination = folder_name / "config_used.txt"
    shutil.copy2(config_source, config_destination)

    print(f"Configuration backed up to: {config_destination}")
    
    # load the dataset
    full_df = pd.read_parquet(cfg['paths']['embeddings'])
    df_known, df_unknown = known_unknown_split(full_df, cfg['unknown_locations'], cfg['bad_individuals'])
    
    # Load list of all burrows (instead make a df with index of unique values from individual_id)
    matrix = pd.read_csv(cfg['paths']['matrix'], index_col='individual_id')
    
    COLL = cfg['collections']
    DICT = {}
    train_df  = pd.DataFrame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # cycles through each year/collection, creating labels, training models and predicting the proceeding year. 
    for index in range(len(COLL) - 1):
    
        # first part generates a train dataset alongside a validation set
        collection = COLL[index]
        test_coll = COLL[index +1]
    
        print(f"\nTraining phase: {collection} | Testing on: {test_coll}")
    
        if index == 0:
    
            train_df = get_train_df(df_known, train_df, collection, DICT, cfg['processing']['min_calls'], cfg['audio_id']['prefix'], cfg['audio_id']['pattern']) 
    
        else:
    
            train_df = append_train_df(df_known, train_df, collection, DICT, cfg['processing']['min_calls'])
    
        # undersamples to create more balanced classes as we have some very large ones
        df = undersample_calls(train_df, cfg['processing']['max_calls'], cfg['processing']['random_seed'])
    
        # splits the dataset into train and validation, and adds 'unknown' calls to the mix
        train_split_df, val_df = train_val_split(df, df_unknown, cfg['processing']['val_size'], cfg['processing']['unk_ratio'])
    
        # generates class weights to help the model counteract class imbalance
        class_weights, known_labels = get_class_weights(df, train_split_df, device)
    
        train_loader = DataLoader(CallDataset(train_split_df), batch_size=cfg['model']['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(CallDataset(val_df), batch_size=cfg['model']['batch_size'])
    
        # Shallow MLP classifier with Entropic Open Set Loss.
        model = AIID_model(cfg['model']['input_dim'], cfg['model']['hidden_dim'], len(known_labels)).to(device)
        criterion = EntropicOpensetLoss(num_of_classes=len(known_labels), device=device, class_weights=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['learning_rate'], weight_decay=cfg['model']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
        best_val_loss = float('inf')
        early_stop_counter = 0
    
        trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=cfg['model']['max_epochs'],
        patience=cfg['model']['patience'],
        save_path=f"models/AIID_model_{collection}.pt"
        )
    
        # Results from the validation set
        results = evaluate_eos_performance(model, val_loader, device, known_labels)
    
        # Creates test dataset with next year's of calls
        test_df = create_test_df(df_known, test_coll, cfg['processing']['min_calls'])

        # model generates probabilities and groups them by individual
        group_results = group_probabilities(model, test_df, known_labels)

        # If there are calls from the burrow for both years, it plots the probabilities for birds returning
        # We want a good split between non-returning (prob = 0) and returning birds (prob = 1)
        plot_return_prob(group_results, test_coll, DICT, folder_name)
        final_df = sort_predictions(
            group_results, 
            cfg['thresholds']['return'], 
            cfg['thresholds']['movement'], 
            cfg['thresholds']['past'], 
            cfg['thresholds']['unknown_label'], 
            test_coll, 
            DICT, 
            cfg['audio_id']['prefix'], 
            cfg['audio_id']['pattern']
        )
        # detailed results, including top 2 predictions and unknown probs for each individual
        result_prob = detailed_probabilities(group_results, test_coll, DICT, folder_name)
        # results by burrow and year, cells populated by audio_id
        new_df, result_cols = get_burrow_results(matrix, test_coll, COLL, DICT, index, folder_name)
        # creates a new id map for analysing next year
        get_combined_map(new_df, result_cols, DICT)
        # results by individual and whether it is present in each year/collection_id
        life_history = get_life_history(new_df, result_cols, folder_name)

if __name__ == "__main__":
    main()