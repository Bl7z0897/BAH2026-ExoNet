import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_kepler_loaders(npz_path, batch_size=64, val_size=0.2, seed=42):
    """
    Safely ingests Phase 1 DR25 .npz matrix structures and outputs 
    clean PyTorch training and validation streaming iterators.
    """
    # 1. Load the compressed file structure
    data = np.load(npz_path)

    # Maps directly to the exact dictionary keys exported by Phase 1
    fluxes = data["flux"]   # Tensor Shape: (2000, 1, 201)
    labels = data["labels"] # Array Shape: (2000,)

    # 2. Split arrays while strictly preserving the 15% class imbalance ratio
    X_train, X_val, y_train, y_val = train_test_split(
        fluxes,
        labels,
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )

    # 3. Convert directly to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 4. Wrap into native PyTorch data-handling utilities
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds   = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
