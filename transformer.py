from ValidationError import ValidationError
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformer models.
    
    Adds positional information to input embeddings to provide sequence order information
    since transformer models don't have inherent notion of order.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of the model embeddings
            dropout (float): Dropout probability
            max_len (int): Maximum sequence length
        """
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
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: Input with positional encoding added and dropout applied
        """
        x = x + self.pe[:, :x.size(1), :]  # [B, T, D]
        return self.dropout(x)


class DecoderOnlyTransformerLayer(nn.TransformerDecoderLayer):
    """
    A decoder-only transformer layer (GPT-style).
    
    This is a modified version of the standard TransformerDecoderLayer that removes
    the cross-attention component, keeping only self-attention and feedforward components.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        """
        Initialize the decoder-only transformer layer.
        
        Args:
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout probability
            activation (function): Activation function to use
            layer_norm_eps (float): Epsilon for layer normalization
            batch_first (bool): If True, batch dimension is first
            norm_first (bool): If True, normalization is applied before attention
            bias (bool): If True, bias is used in linear layers
            device: Device to use
            dtype: Data type to use
        """
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        """
        Forward pass for the decoder-only transformer layer.
        
        This version removes the cross-attention with `memory` and keeps only
        the self-attention and feedforward network components.
        
        Args:
            tgt (Tensor): Input tensor
            memory (Tensor, optional): Not used in decoder-only architecture
            tgt_mask (Tensor, optional): Mask for target sequence
            memory_mask (Tensor, optional): Not used in decoder-only architecture
            tgt_key_padding_mask (Tensor, optional): Padding mask for target sequence
            memory_key_padding_mask (Tensor, optional): Not used in decoder-only architecture
            tgt_is_causal (bool): Whether target attention is causal
            memory_is_causal (bool): Not used in decoder-only architecture
            
        Returns:
            Tensor: Output after self-attention and feedforward
        """
        x = tgt
        if self.norm_first:
            # Apply self-attention with normalization before attention
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # Apply feedforward with normalization before feedforward
            x = x + self._ff_block(self.norm2(x))
        else:
            # Apply self-attention with normalization after attention
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # Apply feedforward with normalization after feedforward
            x = self.norm2(x + self._ff_block(x))

        return x


class TransformerDecoderOnly(nn.Module):
    """
    A decoder-only transformer model (GPT-style).
    
    This model includes token embedding, positional encoding, transformer decoder layers,
    and output projection. It can generate sequences autoregressively.
    """
    
    def __init__(self, vocab_size, d_model, n_head, num_decoder_layers, padding_idx, dim_feedforward=1024):
        """
        Initialize the decoder-only transformer model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the model embeddings
            n_head (int): Number of attention heads
            num_decoder_layers (int): Number of decoder layers
            padding_idx (int): Index of padding token in vocabulary
            dim_feedforward (int): Dimension of the feedforward network
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.posEmbed = PositionalEncoding(d_model)
        decoder_layer = DecoderOnlyTransformerLayer(
            d_model, n_head, batch_first=True, dropout=0.1, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, tgt, tgt_key_padding_mask=None, generating=False, tgt_mask=None, vocab_size=17):
        """
        Forward pass for the decoder-only transformer.
        
        Args:
            tgt (Tensor): Input token indices of shape [batch_size, seq_len]
            tgt_key_padding_mask (Tensor, optional): Mask for padding tokens
            generating (bool): Whether we're in generation mode
            tgt_mask (Tensor, optional): Attention mask for target sequence
            vocab_size (int): Size of vocabulary (used for generation output)
            
        Returns:
            Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # Embed tokens and add positional encoding
        t_emb = self.embed(tgt)
        t_p_emb = self.posEmbed(t_emb)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)
        
        # Pass through decoder
        out_trans = self.decoder(
            t_p_emb, 
            memory=None, 
            tgt_mask=tgt_mask, 
            tgt_is_causal=True, 
            tgt_key_padding_mask=None if generating else tgt_key_padding_mask
        )
        
        # Apply padding mask if not in generation mode
        if not generating:
            out_trans = out_trans * (~tgt_key_padding_mask.unsqueeze(-1))
        
        # Project to vocabulary
        out = self.fc_out(out_trans)
        
        # For generation, limit output to tokens 2-11 (adjust as needed)
        if generating:
            out = out[:, :, 2:12]
            
        return out
    
    def validate_tokens(self, tokens, invalid_tokens, max_attempts=10):
        """
        Validate tokens and resample until valid tokens are obtained or max attempts is reached.
        
        Args:
            tokens (Tensor): Token indices to validate
            invalid_tokens (Tensor): Tensor of invalid token indices
            max_attempts (int): Maximum number of resampling attempts
            
        Returns:
            Tensor: Valid tokens
            bool: Whether validation succeeded
        """
        for attempt in range(max_attempts):
            # Check if any tokens are invalid
            invalid_mask = torch.isin(tokens, invalid_tokens)
            if not torch.any(invalid_mask):
                return tokens, True
                
        return tokens, False
    
    def generate_batch(self, input_tokens, sos_idx, device=None, end_toks_list=None, 
                      max_length=100, temperature=1, batch_size=None, batch_symmetry=False):
        """
        Generate sequences autoregressively.
        
        Args:
            input_tokens (Tensor): Input token indices
            sos_idx (int): Index of the start token
            device (str, optional): Device to use (defaults to self.device)
            end_toks_list (list): List of token indices that indicate end of sequence
            max_length (int): Maximum length of generated sequence
            temperature (float): Temperature for sampling (higher = more random)
            batch_size (int, optional): Batch size (defaults to input_tokens size)
            batch_symmetry (bool): Whether to process each batch item separately
            
        Returns:
            Tensor: Generated sequences
            
        Raises:
            ValidationError: If too many invalid tokens are generated
        """
        self.eval()
        device = self.device if device is None else device
        end_toks_list = [7, 8, 9, 10, 11] if end_toks_list is None else end_toks_list
        
        if batch_size is None:
            batch_size = input_tokens.size(0)
            
        # Initialize sequence with input tokens and start token
        sequence = input_tokens.clone()
        sequence = torch.cat([
            sequence, 
            torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        ], dim=1)
        
        # Track which sequences have reached an end token
        stop_mask = torch.tensor([False] * batch_size, device=device)
        
        # Generate tokens autoregressively
        for i in tqdm(range(max_length), colour="green"):
            with torch.no_grad():
                # Create causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequence.shape[1]).to(self.device)
                
                # Get logits
                if batch_symmetry:
                    # Process each sample independently
                    logits = torch.cat([
                        self(sequence[j:j+1], generating=True, tgt_mask=tgt_mask, vocab_size=10) 
                        for j in range(batch_size)
                    ], dim=0)
                else:
                    # Process all samples in parallel
                    logits = self(sequence, generating=True, tgt_mask=tgt_mask, vocab_size=10)
                
                # Get logits for next token only
                logits = logits[:, -1, :] / temperature
                
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Apply probability cutoff for more stable sampling
            cutoff = 0.001
            mask = probs >= cutoff
            probs = torch.where(mask, probs, torch.tensor(0.0, device=probs.device))
            probs = probs / probs.sum(dim=-1, keepdim=True)
            probs = torch.where(mask, probs, torch.tensor(0.0, device=probs.device))
            
            # Sample next tokens (adding 2 to offset for vocab indexing)
            next_tokens = torch.multinomial(probs, 1) + 2
            
            # For first token, ensure it's not a non-digit token
            if i == 0:
                invalid_first_tokens = torch.tensor([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], device=device)
                nb_attempts = 0
                
                while torch.any(torch.isin(next_tokens, invalid_first_tokens)):
                    # Resample from only digits (indices 0-4 in probs, corresponding to tokens 2-6)
                    next_tokens = torch.multinomial(probs[:, :5], 1) + 2
                    nb_attempts += 1
                    
                    if nb_attempts > 10:
                        raise ValidationError("Too many invalid tokens generated at the beginning of sequence.")
            
            # Ensure year tokens don't appear in generated sequence
            invalid_year_tokens = torch.tensor([12, 13, 14, 15, 16], device=device)
            nb_attempts = 0
            
            while torch.any(torch.isin(next_tokens, invalid_year_tokens)):
                next_tokens = torch.multinomial(probs, 1) + 2
                nb_attempts += 1
                
                if nb_attempts > 10:
                    raise ValidationError("Too many invalid tokens generated during generation.")
            
            # Add next token to sequence
            sequence = torch.cat([sequence, next_tokens], dim=1)
            
            # Check if any sequences have reached an end token
            has_end_tok = torch.isin(next_tokens, torch.tensor(end_toks_list, device=device))
            stop_mask = stop_mask | has_end_tok.flatten()
            
            # If all sequences have ended, stop generation
            if stop_mask.all():
                break
        
        # Add terminal token if needed (reached max length without ending)
        if sequence.shape[1] == input_tokens.size(1) + 1 + max_length:
            col7 = torch.full((batch_size, 1), 7, device=device, dtype=torch.long)
            sequence = torch.cat([sequence, col7], dim=1)
        
        return sequence