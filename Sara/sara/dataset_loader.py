

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_kepler_loaders(npz_path="kepler_processed_dataset.npz", batch_size=64, val_size=0.2, seed=42):
    # 1. Load the compressed file structure
    data = np.load(npz_path)

    # CRITICAL FIX: Mapping keys accurately to the updated archive fields[span_1](start_span)[span_1](end_span)
    fluxes = data["flux"]   # Maps directly to 'flux.npy[span_2](start_span)'[span_2](end_span)
    labels = data["labels"] # Maps directly to 'labels.npy[span_3](start_span)'[span_3](end_span)

    # 2. Here is where the rest of the train_test_split logic sits:
    # This splits the real data while preserving the class imbalance ratio
    X_train, X_val, y_train, y_val = train_test_split(
        fluxes,
        labels,
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )

    # 3. Convert vectors directly to standard 1D-CNN float tensors (N, Channels, Length)[span_4](start_span)[span_4](end_span)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # 4. Wrap into PyTorch iterators matching Tejeswin's loop expectations[span_5](start_span)[span_5](end_span)
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds   = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader