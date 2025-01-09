############################# Regression #######################################   
import json
import os
import math
import random
import sys
import time
from tqdm import tqdm
from typing import Any
from itertools import product

import numpy as np
import pandas as pd
import torch
import argparse

from QuixerfMRI_Regress_Small import get_train_evaluate_regress
from QuixerfMRI_Class_Small import get_train_evaluate_class


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ABCD')
    parser.add_argument("--parcellation", type=str, default='HCP')
    parser.add_argument("--target", type=str, default='nihtbx_fluidcomp_uncorrected')
    parser.add_argument("--model-dir", type=str, default='ABCD_fluidintell_regress_small.pt')  
    parser.add_argument("--lossfunction", type=str, default='MAE')
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()
args=get_args()


batch_size = 32       # Example: 32 sequences per batch
output_dim = 1      # For Binary Classification
input_phenotype = []
input_categorical = []

dataset = args.dataset                   # UKB, ABCD
parcellation = args.parcellation              # HCP, HCP180, Schaefer
target = args.target                    # fluid, nihtbx_fluidcomp_uncorrected
model_dir = args.model_dir    # Directory to save the best model
lossfunction = args.lossfunction        # MAE, MSE
seed = args.seed                        # 2024, 2025, 2026


Set1 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 6, "optimizer": "Adam", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set2 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 6, "optimizer": "AdamW", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set3 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 6, "optimizer": "RMSprop", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set4 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "Adam", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set5 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "AdamW", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set6 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "RMSprop", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set7 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "Adam", "degree": 2, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set8 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "AdamW", "degree": 2, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set9 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "RMSprop", "degree": 2, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set10 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "Adam", "degree": 3, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set11 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "AdamW", "degree": 3, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set12 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 8, "optimizer": "RMSprop", "degree": 3, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set13 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 10, "optimizer": "Adam", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set14 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 10, "optimizer": "AdamW", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set15 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 10, "optimizer": "RMSprop", "degree": 2, "ansatz_layers": 3,
         "n_subjects": 100,
       }

Set16 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 10, "optimizer": "Adam", "degree": 3, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set17 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 10, "optimizer": "AdamW", "degree": 3, "ansatz_layers": 4,
         "n_subjects": 100,
       }

Set18 = {"seed": seed, "output_dim": 1, "dropout": 0.1, "lr": 0.001, 
        "epochs": 10, "restart_epochs": 30000, "batch_size": batch_size, 
        "max_grad_norm": 1.0, "lr_sched": "cos", "wd": 1e-4, "eps": 1e-8, "print_iter": 50,
        "input_phenotype": input_phenotype, "input_categorical": input_categorical,
        "model_dir": model_dir, "dataset": dataset, "parcel_type": parcellation, 
        "target": target, "lossfunction": lossfunction,
        "qubits": 10, "optimizer": "RMSprop", "degree": 3, "ansatz_layers": 4,
         "n_subjects": 100,
       }


hyperparameter_sets = [Set1, Set2, Set3, Set4, Set5, Set6,
                       Set7, Set8, Set9, Set10, Set11, Set12,
                       Set13, Set14, Set15, Set16, Set17, Set18
                      ]


def save_progress(completed_sets, progress_file=f"QuixerTuning/{args.dataset}/{args.parcellation}/Small_{args.target}_{args.lossfunction}_{args.seed}_training_progress.json"):
    """Save the indices of completed hyperparameter sets"""
    with open(progress_file, 'w') as f:
        json.dump({'completed_sets': completed_sets}, f)

def load_progress(progress_file=f"QuixerTuning/{args.dataset}/{args.parcellation}/Small_{args.target}_{args.lossfunction}_{args.seed}_training_progress.json"):
    """Load the indices of completed hyperparameter sets"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed_sets', []))
    return set()


# Load previously completed sets
completed_sets = load_progress()
print(f"Previously completed sets: {sorted(list(completed_sets))}")
    
# Run remaining hyperparameter sets
for idx, hyperparams in enumerate(hyperparameter_sets):
    # Skip if this set was already completed
    if idx in completed_sets:
        print(f"Skipping Set{idx + 1} (already completed)")
        continue
    
    print(f"\nStarting Set{idx + 1}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)
    torch.backends.cudnn.deterministic = True
    
    try:
        # Create a label for each hyperparameter set
        set_label = f"Set{idx + 1}"
        train_evaluate_regress = get_train_evaluate_regress(device, set_label)
        
        # Train and evaluate for each hyperparameter set
        train_evaluate_regress(hyperparams)
        
        # Mark this set as completed
        completed_sets.add(idx)
        save_progress(list(completed_sets))
        print(f"Successfully completed Set{idx + 1}")
        
    except Exception as e:
        print(f"Error in Set{idx + 1}: {str(e)}")
        # Optionally save progress even if there's an error
        save_progress(list(completed_sets))
        raise  # Re-raise the exception to stop execution
    
