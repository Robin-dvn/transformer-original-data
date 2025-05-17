"""
This module contains the main class for the Pommier dataset used for training and validation.
"""

import itertools
from collections import Counter

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split


# PommierDatasetDecoderOnly is a PyTorch Dataset for decoder-only models, tailored for sequence modeling tasks on apple tree data.
class PommierDatasetDecoderOnly(Dataset):
    def __init__(self, dataset_path, token_to_id=None):
        """
        PyTorch Dataset for a decoder-only model.
        Loads a CSV, filters for specific observations and years, tokenizes rows, builds vocabulary, and prepares token IDs.

        Args:
            dataset_path (str): Path to the CSV file containing raw sequences.
            token_to_id (dict, optional): Mapping token -> ID. If None, it will be built automatically.
        """
        # Load the dataset from CSV
        self.dataset = pd.read_csv(dataset_path)
        # Filter for rows where Observation is 'MEDIUM' and Year is 'Y4' or 'Y5'
        self.dataset = self.dataset[((self.dataset["Observation"] == "MEDIUM") & (self.dataset['Year'].isin(['Y4', 'Y5'])))]

        # Tokenize each row on the fly
        self.dataset["tokens"] = self.dataset.apply(lambda row: self.tokenize_row(row), axis=1)

        # Build vocabulary if not provided
        if token_to_id is None:
            self.token_to_id = self.build_vocab(self.dataset["tokens"])
        else:
            self.token_to_id = token_to_id

        # Convert tokens to IDs for each row
        self.dataset["token_ids"] = self.dataset["tokens"].apply(
            lambda tokens: [self.token_to_id[token] for token in tokens]
        )
        # Create a group column for stratified sampling or weighting
        self.dataset["group"] = self.dataset["Observation"] + "_" + self.dataset["Year"]
        groups = self.dataset["group"].tolist()
        group_counts = Counter(groups)
        # Compute inverse frequency weights for each group
        weights = [1.0 / group_counts[g] for g in groups]
        self.weights = weights

    def tokenize_row(self, row):
        """
        Tokenize a row from the dataset.
        Only keeps accepted vocabulary tokens and splits digit strings into individual tokens.
        Example: '123' -> ['1', '2', '3']
        """
        tokens = []
        accepted_voc = ["LARGE", "MEDIUM", "SMALL", "DORMANT", "FLORAL", "Y1", "Y2", "Y3", "Y4", "Y5"]
        for item in row:
            item = str(item).strip()
            if item in accepted_voc:
                tokens.append(item)
            elif item.isdigit():
                tokens.extend(list(item))  # Each digit becomes a token
        return tokens

    def build_vocab(self, token_lists):
        """
        Build the token -> ID mapping, including special tokens <PAD> and <SOS>.
        Args:
            token_lists (list of list of str): List of tokenized sequences.
        Returns:
            dict: Mapping from token to unique integer ID.
        """
        unique_tokens = sorted(set(itertools.chain.from_iterable(token_lists)))
        # Add special tokens
        vocab = {"<PAD>": 0, "<SOS>": 1}
        vocab.update({token: idx + len(vocab) for idx, token in enumerate(unique_tokens)})
        return vocab

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        For each example, build:
          - full_seq = [token1, token2, <SOS>, token3, token4, ...]
          - input_seq  = full_seq[:-1]
          - target_seq = full_seq[1:]
        Loss will be computed only from the token after <SOS>.
        Here, positions 0, 1, and 2 are ignored.
        Args:
            idx (int): Index of the sample.
        Returns:
            input_seq (Tensor): Input sequence for the model.
            target_seq (Tensor): Target sequence for the model.
        """
        token_ids = self.dataset.iloc[idx]["token_ids"]
        # Insert <SOS> after the first two tokens
        full_seq = token_ids[:2] + [self.token_to_id["<SOS>"]] + token_ids[2:]
        input_seq = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(full_seq[1:], dtype=torch.long)
        # print(target_seq)  # Uncomment for debugging
        return input_seq, target_seq


def collate_fn_decoder_only(batch):
    """
    Collate function for DataLoader to pad input and target sequences in a batch.
    Args:
        batch (list of tuples): Each tuple is (input_seq, target_seq).
    Returns:
        inputs (Tensor): Padded input sequences.
        targets (Tensor): Padded target sequences.
    """
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)   # <PAD> = 0
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

if __name__ == "__main__":
    static_dataset = PommierDatasetDecoderOnly("data/all_sequences.csv")
    # Example usage and test of the dataset and dataloader
    VAL_SPLIT = 0.8
    train_size = int(VAL_SPLIT * len(static_dataset))
    val_size = len(static_dataset) - train_size
    # Split the dataset into training and validation sets
    _, val_split = random_split(static_dataset, [train_size, val_size])
    val_loader = DataLoader(val_split, batch_size=2, shuffle=False, collate_fn=collate_fn_decoder_only)
    train_loader = DataLoader(static_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_decoder_only)

    for batch in train_loader:
        inputs, targets = batch
        # Here, you can pass 'batch' to your model for training
        print(inputs, targets)
        break
