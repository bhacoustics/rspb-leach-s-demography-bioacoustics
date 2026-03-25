from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

class CallDataset(Dataset):
    def __init__(self, dataframe):
        self.embeddings = torch.tensor(np.stack(dataframe['embedding'].values), dtype=torch.float32)
        self.labels = torch.tensor(dataframe['idx_label'].values, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.embeddings[idx], self.labels[idx]


class AIID_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.network(x)

class EntropicOpensetLoss:
    """Implementation from AIML-IfI/openset-imagenet"""
    def __init__(self, num_of_classes, device, class_weights, unk_weight=1):
        self.device = device
        self.class_count = num_of_classes
        # Identity matrix for 'One-Hot' encoding knowns
        self.eye = torch.eye(self.class_count).to(device)
        # Uniform distribution for unknowns
        self.unknowns_multiplier = unk_weight / self.class_count
        self.ones = (torch.ones(self.class_count) * self.unknowns_multiplier).to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, logits, target):
        categorical_targets = torch.zeros(logits.shape).to(self.device)
        unk_idx = target < 0
        kn_idx = ~unk_idx
         
        # 1. Handle Knowns: Set target to the specific class (One-Hot)
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        # 2. Handle Unknowns: Set target to a uniform distribution (All classes equal)
        if torch.any(unk_idx):
            categorical_targets[unk_idx, :] = self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
            
        return self.cross_entropy(logits, categorical_targets)

