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


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Transformation 1
    # (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # Inner - Take two consecutive dimesions an group themm. Outer - Transform to complex tensor
    # Transformation 2
    # (seq_len, h, head_dim) -> (B, seq_len, h, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiplication
    # (B, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (B, seq_len, h, head_dim)
    x_rotated = x_complex * freqs_complex
    # Transformation 3
    # (b, seq_len, h, head_dim / 2) -> (b, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Transformation 4
    # (b, seq_len, h, head_dim / 2, 2) -> (b, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (b, seq_len, dim) * (b, seq_len, 1) = (b, seq_len, dim) 
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (dim) * (b, seq_len, dim) = (b, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (b, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]                                             # Basically we introduce a new dimension
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)       # We repeat this dimension number of times
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)     # Then we just flatten it (so we remove again the dimension)
        )


class SelfAttention(nn.Module):
    """
    Self attention mechanism
    Note: compared to the METAs implementation, we removed the paralelization
    here, because I don't have multiple GPUs anyway and the implementatation
    becomes much more simple.
    """
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        # Number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        # Number of heads for the Queries
        self.n_heads_q = args.n_heads

        # Indicates how many times the head of Keys and  Values should be repeated to match the head of the Queries.
        self.n_rep = self.n_heads_q // self.n_kv_heads
        
        # Each head will observe a full sentence but just part of the embedding of each word
        # Idicates dimension of each head
        self.head_dim =  args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (b, 1, dim)

        # Apply the Wq, Wk, and Wv matrices to queries, keys and values
        # (b, 1, dim) -> (b, 1, h_q * head_dim)  h_q=number of query heads
        xq = self.wq(x)
        # (b, 1, dim) -> (b, 1, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # Divde them into corresponding number of heads.
        # (b, 1, h_q * head_dim) -> (b, 1, h_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (b, 1, h_kv * head_dim) -> (b, 1, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (b, 1, h_kv * head_dim) -> (b, 1, h_kv, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (b, 1, h_q, head_dim) -> (b, 1, h_q, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (b, 1, h_kv, head_dim) -> (b, 1, h_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xk

        # Retrieve all the cached keys and values so far
        # (b, seq_len_kv, h_kvm, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_k[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (b, 1, h_q, head_dim) -> (b, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b, h_q, 1, head_dim) @ (b, h_q, head_dim, seq_len_kv) -> (b, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, h_q, 1, seq_len) @ (b, h_q, seq_len_kv, head_dim) -> (b, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # concat heads outputs
        # (b, h_q, 1, head_dim) -> (b, 1, h_q, head_dim) -> (b, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim) -> (b, 1, dim)

        



class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (b, seq_len, dim) + (b, seq_len, dim) -> (b, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


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

