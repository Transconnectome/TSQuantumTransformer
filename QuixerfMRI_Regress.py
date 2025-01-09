import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score, accuracy_score

from QuixerTSModel import QuixerTimeSeries


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.apply(_init_weights)
    

def load_fmri_data(dataset, parcel_type, phenotypes_to_include, target_phenotype, categorical_features):
    """
    Load and combine fMRI data and phenotype data for all subjects.

    Args:
        dataset (str): Type of dataset to use (ABCD, UKB).
        parcel_type (str): Parcellation type for each subject's fMRI data (HCP, HCP180, Schaefer).
        phenotypes_to_include (list): Phenotype columns to include as input features.
        target_phenotype (str): Phenotype column to use as the target.
        categorical_features (list): Columns in phenotypes_to_include that are categorical.

    Returns:
        X (list of tensors): List of input tensors (fMRI + phenotypes) for all subjects.
        y (list): List of target labels for all subjects.
    """
    # Load phenotype data
    if dataset=="ABCD":
        phenotypes = pd.read_csv("ABCD/ABCD_phenotype_total.csv")
        # Drop Subjects with Missing Values
        phenotypes = phenotypes[phenotypes_to_include+["subjectkey", target_phenotype]].dropna()
        subject_ids = phenotypes["subjectkey"].values
    elif dataset=="UKB":
        phenotypes = pd.read_csv("UKB/UKB_phenotype_gps_fluidint.csv")
        # Drop Subjects with Missing Values
        phenotypes = phenotypes[phenotypes_to_include+["eid", target_phenotype]].dropna()
        subject_ids = phenotypes["eid"].values   
    
    # Identify continuous features to normalize
    continuous_features = [col for col in phenotypes_to_include if col not in categorical_features]
    continuous_features.append(target_phenotype)
    
    # Normalize continuous features
    phenotypes[continuous_features] = (
        phenotypes[continuous_features] - phenotypes[continuous_features].mean()
    ) / phenotypes[continuous_features].std()
    
    # Select input phenotypes and target
    input_phenotypes = phenotypes[phenotypes_to_include].values
    target_labels = phenotypes[target_phenotype].values      

    X, y = [], []
    valid_subject_count = 0

    # Load fMRI data for each subject
    for i, subject_id in enumerate(subject_ids):
        if dataset=="ABCD":
            if parcel_type=="HCP":
                fmri_path = f"ABCD/sub-{subject_id}/hcp_mmp1_sub-{subject_id}.npy"
            elif parcel_type=="HCP180":
                fmri_path = f"ABCD/sub-{subject_id}/hcp_mmp1_180_sub-{subject_id}.npy"
            elif parcel_type=="Schaefer":
                fmri_path = f"ABCD/sub-{subject_id}/schaefer_sub-{subject_id}.npy"                
        elif dataset=="UKB":
            if parcel_type=="HCP":
                fmri_path = f"UKB/{subject_id}/hcp_mmp1_{subject_id}.npy"
            elif parcel_type=="HCP180":
                fmri_path = f"UKB/{subject_id}/hcp_mmp1_{subject_id}.npy"
            elif parcel_type=="Schaefer":
                fmri_path = f"UKB/{subject_id}/schaefer_400Parcels_17Networks_{subject_id}.npy"
            
        if not os.path.exists(fmri_path):
            # print(f"Missing fMRI file for subject {subject_id}. Skipping...")
            continue

        fmri_data = np.load(fmri_path)  # Shape: (time_points, brain_regions)

        # Truncate UKB data to 363 time points
        if dataset == "UKB" and fmri_data.shape[0] > 363:
            start_idx = (fmri_data.shape[0] - 363) // 2  # Calculate starting index for truncation
            fmri_data = fmri_data[start_idx:start_idx + 363]
            
        # Compute mean and standard deviation for each brain region
        fmri_mean = fmri_data.mean(axis=0)
        fmri_std = fmri_data.std(axis=0)
        # Avoid division by zero: replace zero std with a small value
        fmri_std[fmri_std == 0] = 1e-8
        # Normalize brain region features
        fmri_data = (fmri_data - fmri_mean) / fmri_std
        fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)

        # Stack phenotype features as additional columns across all time points
        phenotype_tensor = torch.tensor(input_phenotypes[i], dtype=torch.float32).repeat(fmri_tensor.shape[0], 1)
        combined_features = torch.cat((fmri_tensor, phenotype_tensor), dim=1)  # Shape: (time_points, brain_regions + phenotypes)

        X.append(combined_features)
        y.append(target_labels[i])  # Target is one label per subject
        valid_subject_count += 1
        
    print(f"Final sample size (number of subjects): {valid_subject_count}")
    
    return X, y


def split_and_prepare_dataloaders(X, y, batch_size, sequence_length, device):
    """
    Split data into train, validation, and test sets and create DataLoaders with sliding windows.

    Args:
        X (list of tensors): Input data for all subjects (time-series data combined with phenotypes).
        y (list): Target labels for all subjects.
        batch_size (int): Batch size for DataLoader.
        sequence_length (int): Length of each input sequence.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for train, validation, and test sets.
    """
    def create_sequences(data, labels):
        """
        Create fixed-length sequences for each subject using a sliding window.

        Args:
            data (list of tensors): Time-series data for all subjects.
            labels (list): Target labels for all subjects.

        Returns:
            sequences (list of tensors): Sequences of shape (sequence_length, feature_dim).
            sequence_labels (list): Labels for each sequence.
        """
        sequences, sequence_labels = [], []

        for subject_data, label in zip(data, labels):
            num_time_points = subject_data.shape[0]
            for start in range(0, num_time_points - sequence_length + 1, sequence_length):
                seq = subject_data[start:start + sequence_length]  # Fixed-length sequence
                sequences.append(seq)
                sequence_labels.append(label)  # Use the same label for all sequences from this subject

        return sequences, sequence_labels

    # Convert lists to numpy arrays for splitting
    y = np.array(y)

    # Split into train, validation, and test sets
    train_X, temp_X, train_y, temp_y = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    val_X, test_X, val_y, test_y = train_test_split(
        temp_X, temp_y, test_size=0.5, random_state=42
    )

    # Create fixed-length sequences for each split
    train_sequences, train_sequence_labels = create_sequences(train_X, train_y)
    val_sequences, val_sequence_labels = create_sequences(val_X, val_y)
    test_sequences, test_sequence_labels = create_sequences(test_X, test_y)

    # Convert to PyTorch tensors and create DataLoaders
    def create_dataloader(sequences, labels):
        x_tensors = [seq.to(device) for seq in sequences]
        y_tensor = torch.tensor(labels, device=device)
        dataset = TensorDataset(torch.stack(x_tensors), y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader(train_sequences, train_sequence_labels)
    val_loader = create_dataloader(val_sequences, val_sequence_labels)
    test_loader = create_dataloader(test_sequences, test_sequence_labels)

    return train_loader, val_loader, test_loader


def create_model(
    hyperparams: dict[str, Any], device: torch.device, sequence_length: int, feature_dim: int
) -> torch.nn.Module:
    model = QuixerTimeSeries(
        n_qubits=hyperparams["qubits"],
        n_timesteps=sequence_length,
        degree=hyperparams["degree"],
        n_ansatz_layers=hyperparams["ansatz_layers"],
        feature_dim=feature_dim,
        output_dim=hyperparams["output_dim"],
        dropout=hyperparams["dropout"],
        device=device,
    )
    return model


def train_epoch(
    model: torch.nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
):
    model.train()
    epoch_loss = 0

    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        yhat, norm_avg = model(x)

        loss = criterion(yhat.squeeze(), y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(
    model: torch.nn.Module,
    iterator: DataLoader,
    criterion,
):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for x, y in tqdm(iterator):
            yhat, _ = model(x)
            loss = criterion(yhat.squeeze(), y)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    tuning_set,    
):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
        )
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
        )    
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
        )

    scheduler = None
    if hyperparams["lr_sched"] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    if hyperparams["lossfunction"]=="MAE":
        criterion = torch.nn.L1Loss()  # MAE loss function for regression tasks
    elif hyperparams["lossfunction"]=="MSE":
        criterion = torch.nn.MSELoss()  # MSE loss function for regression tasks

    # Lists to store metrics
    train_metrics, valid_metrics, test_metrics = [], [], []
    
    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            hyperparams["max_grad_norm"],
            scheduler,
        )

        valid_loss = evaluate(model, val_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if hyperparams["model_dir"]==None:
                torch.save(model.state_dict(), f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/best_time_series_model.pt")
            else:
                torch.save(model.state_dict(), f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/{hyperparams['model_dir']}_{hyperparams['lossfunction']}_{hyperparams['seed']}_{tuning_set}.pt")
                
        # Append train and validation metrics for each epoch
        train_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss})
        valid_metrics.append({'epoch': epoch + 1, 'valid_loss': valid_loss})
        
        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}")

    if hyperparams["model_dir"]==None:
        model.load_state_dict(torch.load(f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/best_time_series_model.pt", weights_only=True))
    else:
        model.load_state_dict(torch.load(f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/{hyperparams['model_dir']}_{hyperparams['lossfunction']}_{hyperparams['seed']}_{tuning_set}.pt", weights_only=True))
    test_loss = evaluate(model, test_loader, criterion)

    # Save the test metrics after training
    test_metrics.append({'epoch': hyperparams['epochs'], 'test_loss': test_loss})    
    print(f"Test Loss: {test_loss:.3f}")

    # Combine all metrics into a pandas DataFrame
    metrics = []
    for epoch in range(hyperparams['epochs']):
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics[epoch]['train_loss'],
            'valid_loss': valid_metrics[epoch]['valid_loss'],
            'test_loss': test_metrics[0]['test_loss'],
        })
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Save to CSV
    csv_filename = f"QuixerTuning/{hyperparams['dataset']}/{hyperparams['parcel_type']}/{hyperparams['target']}_{hyperparams['lossfunction']}_{hyperparams['seed']}_{tuning_set}.csv"
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")
    
    return test_loss


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate_regress(device: torch.device, tuning_set) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        seed(parameterization["seed"])

        sequence_length = 363        
        # if parameterization["dataset"] == "ABCD":
        #     sequence_length = 363
        # elif parameterization["dataset"] == "UKB":
        #     sequence_length = 490
            
        X, y = load_fmri_data(
            parameterization["dataset"],
            parameterization["parcel_type"],
            parameterization["input_phenotype"],
            parameterization["target"],       
            parameterization["input_categorical"],
        )
        
        train_loader, val_loader, test_loader = split_and_prepare_dataloaders(
            X, y,
            parameterization["batch_size"],
            sequence_length,
            device,
        )

        feature_dim = train_loader.dataset[0][0].shape[-1]  
        model = create_model(parameterization, device, sequence_length, feature_dim)

        init_weights(model)
        model = model.to(device)

        test_loss = train_cycle(
            model,
            parameterization,
            device,
            train_loader,
            val_loader,
            test_loader,
            tuning_set,
        )

        return test_loss

    return train_evaluate




