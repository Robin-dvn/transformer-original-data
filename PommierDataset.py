from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import itertools

from enums import Observation
import numpy as np
from sequences import terminal_fate, _generate_random_draw_sequence
from torch.utils.data import random_split



class PommierDatasetDecoderOnly(Dataset):
    def __init__(self, dataset_path, token_to_id=None):
        """
        Dataset PyTorch pour un modèle Décodeur-only.

        Args:
            dataset_path (str): Chemin du fichier CSV contenant les séquences brutes.
            token_to_id (dict, optional): Mapping token -> ID. S'il est None, il sera construit.
        """
        self.dataset = pd.read_csv(dataset_path)
        self.dataset = self.dataset[((self.dataset["Observation"] == "MEDIUM") & (self.dataset['Year'].isin(['Y4', 'Y5'])))]

        # Tokenisation à la volée
        self.dataset["tokens"] = self.dataset.apply(lambda row: self.tokenize_row(row), axis=1)

        # Construction du vocabulaire (si non fourni)
        if token_to_id is None:
            self.token_to_id = self.build_vocab(self.dataset["tokens"])
        else:
            self.token_to_id = token_to_id

        # Conversion des tokens en IDs
        self.dataset["token_ids"] = self.dataset["tokens"].apply(
            lambda tokens: [self.token_to_id[token] for token in tokens]
        )
        self.dataset["group"] = self.dataset["Observation"] + "_" + self.dataset["Year"]
        groups = self.dataset["group"].tolist()
        group_counts = Counter(groups)
        weights = [1.0 / group_counts[g] for g in groups]
        self.weights = weights
    def tokenize_row(self, row):
        """Tokenise une ligne du dataset."""
        tokens = []
        accepted_voc = ["LARGE", "MEDIUM", "SMALL", "DORMANT", "FLORAL", "Y1", "Y2", "Y3", "Y4", "Y5"]
        for item in row:
            item = str(item).strip()
            if item in accepted_voc:
                tokens.append(item)
            elif item.isdigit():
                tokens.extend(list(item))  # Chaque chiffre devient un token
        return tokens

    def build_vocab(self, token_lists):
        """Construit le mapping token -> ID en incluant les tokens spéciaux."""
        unique_tokens = sorted(set(itertools.chain.from_iterable(token_lists)))
        # Ajout des tokens spéciaux
        vocab = {"<PAD>": 0, "<SOS>": 1}
        vocab.update({token: idx + len(vocab) for idx, token in enumerate(unique_tokens)})
        return vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Pour chaque exemple, on construit :
          - full_seq = [token1, token2, <SOS>, token3, token4, ...]
          - input_seq  = full_seq[:-1]
          - target_seq = full_seq[1:]
        La perte sera calculée uniquement à partir du token situé après <SOS>.
        Ici, on ignore les positions 0, 1 et 2.
        """
        token_ids = self.dataset.iloc[idx]["token_ids"]
        full_seq = token_ids[:2] + [self.token_to_id["<SOS>"]] + token_ids[2:]
        input_seq = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(full_seq[1:], dtype=torch.long)
        # print(target_seq)


        return input_seq, target_seq


def collate_fn_decoder_only(batch):
    inputs, targets= zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)   # <PAD> = 0
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

if __name__ == "__main__":
    VAL_SPLIT = 0.8
    vocab_to_id ={'<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, 'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11, 'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16}


    static_dataset = PommierDatasetDecoderOnly("data/all_sequences.csv")
    # dataset = DecoderOnlyDynamicPommierDataset(vocab_to_id, 10000, 4, 70)
    train_size = int(VAL_SPLIT * len(static_dataset))
    val_size = len(static_dataset) - train_size
    _, val_split = random_split(static_dataset, [train_size, val_size])
    val_loader = DataLoader(val_split, batch_size=2, shuffle=False, collate_fn=collate_fn_decoder_only)
    train_loader = DataLoader(static_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_decoder_only)

    for batch in train_loader:
        inputs, targets, masks = batch
        # Ici, vous pouvez passer 'batch' à votre modèle pour l'entraînement
        print(inputs, targets)
        break
