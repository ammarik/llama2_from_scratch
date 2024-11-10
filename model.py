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

