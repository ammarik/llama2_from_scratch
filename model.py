import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32                   # Number of heads for the queries (Q)
    n_kv_heads: Optional[int] = None    # Number of heads for the K a V
    vocab_size: int = -1                # This will be set wen we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_freqs(head_dim: int, seq_len: int, device: str, theta: Optional[float]=10000.0):
    assert head_dim % 2 == 0, 'DImension must be divisible by 2'
    # Build the theta parameters
    # According to the formula: theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2 ... dim/2]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the 'm' parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply by each position using the outer product
    # Shape: (seq_len) * (hed_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

 
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, 'Vocab size must be set'

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # Precompute the frequencies of the rotary positional encodings
        self.freqs_complex = precompute_theta_pos_freqs(args.dim // args.n_heads, args.max_seq_len * 2, device=args.device)
  
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (b, seq_len)
        batch_size, seq_len = tokens.shape
        # Note: This model is intended only for inferencing not training because for training
        # of course we need to not have the KV cache and we need to be able to process multiple
        # tokens, but our goal is actually to use the pre-trained LLaMA weights.
        assert seq_len == 1, 'Only one token at a time can be processed'

        # (b, seq_len) -> (b, seq_len, dim) # dim is 4096 for base 7B model, buy depending on the model size it can be different 
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions  [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output

