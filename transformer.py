
from ValidationError import ValidationError
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # [B, T, D]


        return self.dropout(x)


class DecoderOnlyTransformerLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        """
        Version Decoder-Only :
        - Supprime le Cross-Attention avec `memory`
        - Garde uniquement le Self-Attention causale et le Feed-Forward Network (FFN)
        """

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # 🚀 Cross-Attention supprimé 🚀
            x = x + self._ff_block(self.norm2(x))  # FFN
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # 🚀 Cross-Attention supprimé 🚀
            x = self.norm2(x + self._ff_block(x))  # FFN

        return x



class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_decoder_layers, padding_idx,dim_feedforward=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.posEmbed = PositionalEncoding(d_model)
        decoder_layer = DecoderOnlyTransformerLayer(d_model, n_head, batch_first=True, dropout=0.1,dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, tgt: torch.Tensor, tgt_key_padding_mask: torch.Tensor = None,generating= False,tgt_mask= None,vocab_size=17):
        t_emb = self.embed(tgt)
        t_p_emb = self.posEmbed(t_emb)
        # print(t_p_emb)

        # print("Embeddings avant d'entrer dans le décodeur :", t_p_emb)

        tgt_mask_ = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device) if tgt_mask == None else tgt_mask
        # print(tgt_mask)
        # memory = torch.zeros(tgt.shape[0], tgt.shape[1], t_p_emb.shape[-1], device=self.device)

        out_trans = self.decoder(
            t_p_emb,memory=None, tgt_mask=tgt_mask_, tgt_is_causal=True, tgt_key_padding_mask=None if generating else tgt_key_padding_mask
        )
        # print(out_trans[:,:3,:])

        if not generating : out_trans = out_trans * (~tgt_key_padding_mask.unsqueeze(-1))  # Masque les positions padding

        out = self.fc_out(out_trans)
        if generating:
            out = out[:,:,2:12]
            # print(out.size())
        return out
    def generate_batch(self, input_tokens, sos_idx, device, end_toks_list, max_length=100, temperature=1, batch_size=None, batch_symmetry=False):
        self.eval()
        # self.device = 'cpu'
        device = self.device
        # input_tokens = input_tokens.cpu()
        # self.cpu()
        if batch_size is None:
            batch_size = input_tokens.size(0)

        sequence = input_tokens.clone()
        sequence = torch.cat([sequence, torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)], dim=1)
        stop_mask = torch.tensor([False] * batch_size, device=device)

        for i in tqdm(range(max_length), colour="green"):
            with torch.no_grad():
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequence[0:1].shape[1]).to(self.device)
                if batch_symmetry:


                    logits = torch.cat([
                        self(sequence[j:j+1], generating=True,tgt_mask = tgt_mask,vocab_size = 10) for j in range(batch_size)
                    ], dim=0)
                    print(logits.size())

                else:
                    logits = self(sequence, generating=True,tgt_mask = tgt_mask,vocab_size=10)

                if i == 0:
                    diffs = torch.abs(logits - logits[0])
                    max_diff = diffs.max()
                    all_equal = torch.all(diffs < 1e-6)
                    print(f"[DEBUG i=0] Diff max logits ligne par ligne : {max_diff.item():.6f}")
                    print(f"[DEBUG i=0] Toutes les lignes identiques ? {all_equal.item()}")

                logits = logits[:, -1, :] / temperature


            probs = F.softmax(logits, dim=-1)
            cutoff = 0.001
            mask = probs >= cutoff
            probs = torch.where(mask, probs, torch.tensor(0.0, device=probs.device))
            probs = probs / probs.sum(dim=-1, keepdim=True)
            probs = torch.where(mask, probs, torch.tensor(0.0, device=probs.device))
            # print(probs[:5])

            # print(probs[-5:])
            next_tokens = torch.multinomial(probs, 1)+2


            if i == 0:
                nb_max = 0
                invalid_tokens = torch.tensor([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], device=device)

                while torch.any(torch.isin(next_tokens, invalid_tokens)):
                    mask_invalid = torch.isin(next_tokens, invalid_tokens).squeeze()  # [batch_size]
                    for idx in torch.nonzero(mask_invalid, as_tuple=False):
                        token = next_tokens[idx].item()

                        prob = probs[idx, token-2].item()
                        # print(f"[Token invalide] ligne {idx.item()} : token={token}, proba={prob:.6f}")

                    next_tokens = torch.multinomial(probs[:,:5], 1)+2
                    nb_max += 1
                    if nb_max > 10:
                        raise ValidationError("Trop de tokens non valides générés en début de séquence.")


            nb_max = 0
            invalid_tokens = torch.tensor([12, 13, 14, 15, 16], device=device)
            while torch.any(torch.isin(next_tokens, torch.tensor([ 12, 13, 14, 15, 16], device=device))):
                mask_invalid = torch.isin(next_tokens, invalid_tokens).squeeze()  # [batch_size]
                for idx in torch.nonzero(mask_invalid, as_tuple=False):
                    token = next_tokens[idx].item()
                    prob = probs[idx, token-2].item()
                    # print(f"[Token invalide] ligne {idx.item()} : token={token}, proba={prob:.6f}")
                next_tokens = torch.multinomial(probs, 1)+2
                nb_max += 1
                if nb_max > 10:
                    raise ValidationError("Trop de tokens non valides générés durant la génération.")

            sequence = torch.cat([sequence, next_tokens], dim=1)
            has_end_tok = torch.isin(next_tokens, torch.tensor(end_toks_list, device=device))
            stop_mask = stop_mask | has_end_tok.flatten()
            if stop_mask.all():
                break

        if sequence.shape[1] == input_tokens.size(1) + 1 + max_length:
            col7 = torch.full((batch_size, 1), 7, device=device, dtype=torch.long)
            sequence = torch.cat([sequence, col7], dim=1)

        return sequence
