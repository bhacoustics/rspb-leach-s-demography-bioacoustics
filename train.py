import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                max_epochs=10, patience=5, save_path="AIID_model.pt"):
    
    print(f"Training on {device}...")
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for feats, targs in train_loader:
            feats, targs = feats.to(device), targs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), targs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feats, targs in val_loader:
                feats, targs = feats.to(device), targs.to(device)
                val_loss += criterion(model(feats), targs).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        # Step the scheduler based on validation loss
        scheduler.step() 
        
        print(f"Epoch {epoch:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # --- Checkpoint & Early Stopping ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}\n")
                break

    # Load the best weights before finishing
    model.load_state_dict(torch.load(save_path))
    return model

def calculate_oscr(all_logits, all_targets, OOD_INT=-1):
    """
    Vaze et al. implementation for OSCR.
    all_logits: (N, num_classes) - The raw scores/probabilities for known classes
    all_targets: (N,) - Labels where OOD_INT represents unknown samples
    """
    if isinstance(all_logits, torch.Tensor):
        all_logits = all_logits.cpu().numpy()
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()

    # Split into known (ID) and unknown (OOD)
    id_mask = all_targets != OOD_INT
    ood_mask = all_targets == OOD_INT
    
    id_logits = all_logits[id_mask]
    id_targets = all_targets[id_mask]
    ood_logits = all_logits[ood_mask]

    # Classify known samples
    preds = id_logits.argmax(axis=1)
    correct = (preds == id_targets)
    
    # We only care about the max logit (confidence) for correctly classified samples
    # and the max logit for OOD samples (to see if they are falsely accepted)
    correctly_classified_id_scores = id_logits[correct].max(axis=1)
    ood_scores = ood_logits.max(axis=1)

    n = len(all_logits)
    CCR = np.zeros(n + 1)
    FPR = np.zeros(n + 1)

    # Generate thresholds across the range of scores
    thetas = np.linspace(all_logits.min(), all_logits.max(), n)
    
    for i, theta in enumerate(thetas):
        # Correct Classification Rate: Correct ID and Score > Threshold
        CC = (correctly_classified_id_scores > theta).sum()
        # False Positive Rate: OOD sample and Score > Threshold
        FP = (ood_scores > theta).sum()
        
        CCR[i] = CC / len(id_logits)
        FPR[i] = FP / len(ood_logits)

    # Sort for trapezoidal integration
    ROC = sorted(zip(FPR, CCR), reverse=True)
    OSCR = 0
    for j in range(len(ROC) - 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0
        OSCR += h * w

    return OSCR

def evaluate_eos_performance(model, val_loader, device, known_labels, threshold=0.5):
    model.eval()
    
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for feats, targs in val_loader:
            feats = feats.to(device)
            # We need the raw logits (or probs) for the OSCR function
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)
            
            all_logits.append(probs.cpu()) # Using probs here as they are bounded 0-1
            all_targets.append(targs.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # 1. OSCR Score (Passing all_logits and all_targets)
    oscr_score = calculate_oscr(all_logits, all_targets, OOD_INT=-1)

    # 2. General Predictions for F1 and Accuracy
    # Note: preds only looks at known classes (0-98)
    preds = np.argmax(all_logits, axis=1)
    max_probs = np.max(all_logits, axis=1)
    
    # 3. Known Accuracy (Ignore unknowns)
    known_mask = all_targets >= 0
    known_acc = accuracy_score(all_targets[known_mask], preds[known_mask])

    # 4. Open-Set AUROC
    binary_true = (all_targets >= 0).astype(int)
    auroc_score = roc_auc_score(binary_true, max_probs)

    # 5. Summary Metrics (Macro and Weighted)
    # Apply threshold to determine if we predict -1 (Unknown)
    final_preds = np.where(max_probs < threshold, -1, preds)
    
    report = classification_report(all_targets, final_preds, 
                                   labels=[-1] + list(range(len(known_labels))),
                                   output_dict=True, zero_division=0)

    print(f"Model results from validation set:\n")
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 33)
    print(f"{'OSCR Score':<20} | {oscr_score:.4f}")
    print(f"{'Open-Set AUROC':<20} | {auroc_score:.4f}")
    print(f"{'Known Accuracy':<20} | {known_acc:.4f}")
    print(f"{'Unknown F1-Score':<20} | {report['-1']['f1-score']:.4f}")
    print("-" * 33)
    print(f"{'Macro F1-Score':<20} | {report['macro avg']['f1-score']:.4f}")
    print(f"{'Weighted F1-Score':<20} | {report['weighted avg']['f1-score']:.4f}")


    return {
        'oscr': oscr_score,
        'known_acc': known_acc,
        'auroc': auroc_score,
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg'],
        'unknown_f1': report['-1']['f1-score']
    }