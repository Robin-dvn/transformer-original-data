"""
Pipeline module for training, generation, and validation of Transformer models.
This module contains the main functions to manage the complete learning workflow.
"""


# Standard library imports
import json
import subprocess
from datetime import datetime
from pathlib import Path
from time import time

# Third-party imports
import numpy as np
import optuna
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import wandb

# Local imports
from EarlyStopping import EarlyStopping
from PommierDataset import (
    PommierDatasetDecoderOnly,
    collate_fn_decoder_only,
)
from transformer import TransformerDecoderOnly

from ValidationError import ValidationError, GPUOutOfMemoryError


def model_size_mb(model):
    """
    Calculate the size of a PyTorch model in megabytes.

    Args:
        model: The PyTorch model to calculate the size for.

    Returns:
        float: Model size in megabytes.
    """
    total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return total_size / (1024 ** 2)


def create_config_file(file_path, config_dict):
    """
    Create a JSON configuration file from a dictionary.

    Args:
        file_path (str or Path): Path to the file where the configuration will be saved.
        config_dict (dict): Dictionary containing configuration parameters.

    Raises:
        IOError: If the file cannot be created or written.
        TypeError: If config_dict is not a dictionary.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("config_dict must be a dictionary")

    # Convert Path objects to strings for JSON serialization
    config_dict_serializable = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in config_dict.items()
    }

    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(config_dict_serializable, json_file, indent=4)
    except IOError as e:
        raise IOError(f"Could not create configuration file: {e}") from e


def train_decoder_only(config_dict, trial=None):
    """
    Train a decoder-only transformer model according to the provided configuration.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.
        trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization.

    Returns:
        tuple: (trained model, experiment folder path, final validation loss, wandb run)
    """

    # Extract parameters from config_dict
    dataset_path = config_dict['dataset_path']
    seed = config_dict['seed']
    batch_size = config_dict['batch_size']
    val_split = config_dict['val_split']
    vocab_size = config_dict['vocab_size']
    padding_idx = config_dict['padding_idx']
    n_head = config_dict['n_head']
    d_model = config_dict['d_model']
    nb_layers = config_dict['nb_layers']
    lr = config_dict['lr']
    nb_epoch = config_dict['nb_epoch']
    dim_feedforward = config_dict['dim_feedforward']
    dynamic = config_dict['dynamic']
    scheduler_config = config_dict['scheduler']
    early_stopping_config = config_dict['early_stopping']
    continue_training = config_dict['continue_training']
    checkpoint_path = config_dict['checkpoint_path']
    auto_precision = config_dict['auto_precision']

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate a timestamp for the experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = (
        f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_TS-{timestamp}"
    )
    experiment_path = Path("experiments") / exp_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    print(str(experiment_path / "config.json"))
    create_config_file(experiment_path / "config.json", config_dict)

    # Dataset creation
    dataset = PommierDatasetDecoderOnly(dataset_path)




    train_size = int(val_split * len(dataset))
    val_size = len(dataset) - train_size
    trian_spit, val_split = random_split(dataset, [train_size, val_size])

    # Recalculate groups only on the training split
    train_groups = [dataset.dataset.iloc[i]["group"] for i in trian_spit.indices]

    from collections import Counter
    group_counts = Counter(train_groups)
    train_weights = [1.0 / group_counts[g] for g in train_groups]

    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(trian_spit), replacement=True)
    train_loader = DataLoader(trian_spit, sampler=sampler, shuffle=False, batch_size=batch_size, collate_fn=collate_fn_decoder_only)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_decoder_only)

    # Model creation
    model = TransformerDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        num_decoder_layers=nb_layers,
        padding_idx=padding_idx,
        dim_feedforward=dim_feedforward
    )
    model.to(device)
    if continue_training:
        model.load_state_dict(torch.load(checkpoint_path,weights_only = True)["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Scheduler creation (if applicable)
    scheduler = None
    if scheduler_config['name'] == "cyclical":
        scheduler = CyclicLR(optimizer, **scheduler_config['params'])
    elif scheduler_config['name'] == "ReduceOnPlatau":
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_config['params'])

    # Early stopping setup (if applicable)
    early_stopping = None
    if early_stopping_config['name'] == "patience":
        early_stopping = EarlyStopping(**early_stopping_config['params'])

    # Calculate the number of trainable parameters and model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = model_size_mb(model)

    # Initialize wandb for experiment tracking
    wandb_config = {
        "learning_rate": lr,
        "val_split": config_dict['val_split'],
        "architecture": exp_name,
        "dataset": "100 sample de chaque type",
        "batch_size": batch_size,
        "dimension_model": d_model,
        "number_of_heads": n_head,
        "epochs": nb_epoch,
        "dynamic": dynamic,
        "num_layers": nb_layers,
        "num_params": num_params,
        "dim_feedforward": dim_feedforward,
        "scheduler": scheduler_config['name'],
        "scheduler_params": (
            scheduler_config['params'] if scheduler_config['name'] != "None"
            else None
        ),
        "early_stopping": early_stopping_config['name'],
        "early_stopping_params": (
            early_stopping_config['params'] if early_stopping_config['name'] != "None"
            else None
        ),
        "auto_precision": auto_precision

    }
    wandb.init(
        name=exp_name,
        project="Topologie-Pommiers-Original-Data",
        config=wandb_config,
        mode="offline"
    )

    print(f"Number of parameters : {num_params:,}")
    print(f"The model occupies approximately {size_mb:.2f} MB in memory.")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    # Setup GradScaler if auto_precision is enabled
    scaler = torch.amp.GradScaler(device=device) if auto_precision else None

    # Training loop
    global_batch = 0
    best_val_loss = float('inf')
    val_losses = []  # List to store validation losses
    for epoch in tqdm(range(nb_epoch), colour="green"):
        model.train()
        total_train_loss = 0
        for input_seq, target_seq in tqdm(
            train_loader,
            desc=f"Epoch {epoch} - Train",
            colour="green"
        ):
            try:
                if device == "cuda":
                    mem_alloc = torch.cuda.memory_allocated(device) / 1024**2  # in MB
                    # wandb.log({"gpu_memory_allocated_MB": mem_alloc}, step=global_batch)
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                padding_mask = (input_seq == 0).to(torch.bool).to(model.device)

                if auto_precision:
                    with torch.amp.autocast(device_type=device):
                        logits = model(input_seq, padding_mask)
                        logits_trim = logits[:, 2:, :]
                        targets_trim = target_seq[:, 2:]
                        logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                        target_flat = targets_trim.reshape(-1)
                        loss = criterion(logits_flat, target_flat)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(input_seq, padding_mask)
                    logits_trim = logits[:, 2:, :]
                    targets_trim = target_seq[:, 2:]
                    logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                    target_flat = targets_trim.reshape(-1)
                    loss = criterion(logits_flat, target_flat)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if scheduler is not None and scheduler_config['name'] == "cyclical":
                    scheduler.step()

                total_train_loss += loss.item()

                # Log the learning rate at each batch
                current_lr = optimizer.param_groups[0]['lr']
                # wandb.log({"batch_learning_rate": current_lr}, step=global_batch)
                global_batch += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    raise GPUOutOfMemoryError(f"GPU memory error during training. Configuration: {config_dict}")
                raise e

        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for input_seq, target_seq in tqdm(
                val_loader,
                desc=f"Epoch {epoch} - Val",
                colour="blue"
            ):
                try:
                    input_seq = input_seq.to(device)
                    target_seq = target_seq.to(device)
                    padding_mask = (input_seq == 0).to(torch.bool).to(model.device)
                    logits = model(input_seq, padding_mask)
                    logits_trim = logits[:, 2:, :]
                    targets_trim = target_seq[:, 2:]
                    logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                    target_flat = targets_trim.reshape(-1)
                    loss = criterion(logits_flat, target_flat)
                    total_eval_loss += loss.item()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        raise GPUOutOfMemoryError(f"GPU memory error during validation. Configuration: {config_dict}")
                    raise e

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_eval_loss / len(val_loader)

        val_losses.append(avg_val_loss)  # Store validation loss

        # If in an Optuna trial, report validation loss at each epoch
        if trial is not None:
            trial.report(avg_val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Update best validation loss
            best_val_loss = min(best_val_loss, avg_val_loss)

        wandb.log({
            "train_loss_epochs": avg_train_loss,
            "val_loss_epochs": avg_val_loss
        })
        tqdm.write(
            f"[INFO] Epoch {epoch} : train loss unweighted = {avg_train_loss:.4f}, "
            f"val loss unweighted = {avg_val_loss:.4f}"
        )
        if scheduler is not None and scheduler_config['name'] == "ReduceOnPlatau":
            scheduler.step(avg_val_loss)

        if early_stopping:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, experiment_path / "model_state.pth")

    # Calculate the average over the last 20 epochs
    last_20_epochs_val_loss = sum(val_losses[-20:]) / min(20, len(val_losses))

    # If in an Optuna trial, use the average of the last 20 epochs
    final_val_loss = last_20_epochs_val_loss

    return model, experiment_path, final_val_loss, wandb.run



def find_wandb_run_path(run_id):
    """
    Find the path to the wandb folder containing the specified run ID.

    Args:
        run_id: The wandb run ID.

    Returns:
        str: The full path to the run folder.
    """
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        raise FileNotFoundError("The wandb directory does not exist")

    for run_dir in wandb_dir.iterdir():
        if run_dir.is_dir() and str(run_id) in run_dir.name:
            return str(run_dir)

    raise FileNotFoundError(f"No folder found containing ID {run_id}")


def train_generate_validate_pipeline(config_dict, trial=None, sync_wandb=False):
    """
    Pipeline to train, generate, and validate a model using a configuration dictionary.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.
        trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization.
        sync_wandb (bool, optional): If True, synchronizes wandb data online at the end of the run.

    Returns:
        Validator: Instance of the Validator class used to generate and validate data.
    """
    # Train the model
    st = time()
    model, experiment_path, final_val_loss, wandb_run = train_decoder_only(config_dict, trial)
    et = time()
    print(f"[INFO] Training time in hours: {(et-st)/3600}")

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device=device)
    vocab_to_id = {
        '<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
        'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11,
        'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16
    }

    st = time()
    try:
        generate_data(model, device, vocab_to_id, 10000, experiment_path / "generated_dataset.csv", end_toks_list=[7, 8, 9, 10, 11])
    except ValidationError as e:
        print(f"[ERROR] {e}")
        return None
    et = time()
    print(f"[INFO] Generation time in seconds: {et-st}")

    # Close wandb run with or without online synchronization
    if sync_wandb:
        # First, finish the run in offline mode
        wandb_run.finish(quiet=True)
        # Then, synchronize online using the unique run ID
        print("[INFO] Synchronizing wandb data online...")
        try:
            run_path = find_wandb_run_path(wandb_run.id)
            subprocess.run(
                ["wandb", "sync", run_path],
                check=True,
                capture_output=True,
                text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[WARNING] Error during wandb synchronization: {e}")
    else:
        wandb_run.finish(quiet=True)

    # return metrics["final_val"]
    return


def generate_data(model, device, token_to_id, nb_samples, output_path, end_toks_list):
    """
    Generate data using the Transformer model and save to CSV.

    Args:
        model: The trained Transformer model.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        token_to_id (dict): Mapping from token to id.
        nb_samples (int): Number of samples to generate per (type, year) pair.
        output_path (str or Path): Path to save the generated CSV.
        end_toks_list (list): List of end token ids.
    """
    import torch
    import pandas as pd
    from tqdm import tqdm

    id_to_token = {v: k for k, v in token_to_id.items()}
    sequences_generees = []
    decoder_only = True
    for type in tqdm(range(9, 11)):
        for year in range(12, 17):
            if nb_samples > 1000:
                for i in range(0, nb_samples, 1000):
                    batch_size = min(1000, nb_samples - i)
                    start_seq = torch.tensor([[type, year]] * batch_size, device=device)
                    generated_seq = model.generate_batch(start_seq, 1, device, end_toks_list, batch_size=int(batch_size))
                    if not decoder_only:
                        sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 1:]), dim=1).to('cpu').tolist())
                    else:
                        sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 3:]), dim=1).to('cpu').tolist())
            else:
                start_seq = torch.tensor([[type, year]] * nb_samples, device=device)
                generated_seq = model.generate_batch(start_seq, 1, device, end_toks_list, batch_size=nb_samples)
                if not decoder_only:
                    sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 1:]), dim=1).to('cpu').tolist())
                else:
                    sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 3:]), dim=1).to('cpu').tolist())

    print(f"[INFO] Generated {len(sequences_generees)} sequences")
    print(f"[INFO] converting to dataset: ")
    data_generated = []
    for seq in tqdm(sequences_generees):
        datasetform = []
        digits = ""
        for item in seq:
            if item in id_to_token:
                if id_to_token[item].isdigit():
                    digits += id_to_token[item]
                    continue
                if digits != "":
                    datasetform.append(digits)
                datasetform.append(id_to_token[item])
                digits = ""
                if item in end_toks_list and len(datasetform) != 1:
                    break
        data_generated.append(datasetform)
    df = pd.DataFrame(data_generated, columns=["Observation", "Year", "Sequence", "Terminal Fate"])
    print(f"[INFO] Saving to {output_path}")
    df.to_csv(output_path, index=False)
