import json
import time
from pathlib import Path
from typing import Optional

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    
    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0, 'No checkpoint files found!'
            chk_path = checkpoints[0]
            print(f'Loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f'Loaded checkpoint in {time.time() - prev_time:.2f}s')

        with open(Path(checkpoint_dir) / 'params.json', 'r') as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time() - prev_time:.2f}s')
        
        return LLaMA(model, tokenizer, model_args)
    
if __name__ == '__main__':
    torch.manual_seed(0)

  # Select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif  torch.backends.mps.is_available():
        device = 'mps'

    device = 'cpu'
    #device = torch.device(device)
    print(f'Using device: {device}')
    

    # prompts = [
    #     "Simply put, the theory of relativity states that ",
    #     "If Google was an Italian company founded in Milan, it would",
    #     # Few shot promt
    #     """Translate English to French:
        
    #     sea otter => loutre de mer
    #     peppermint => menthe poivrée
    #     plush girafe => girafe peluche
    #     cheese =>""",
    #     # Zero shot prompt
    #     """Tell me if the following person is actually Doraemon disguised as human:
    #     Name: Umar Jamil
    #     Decision: 
    #     """
    # ]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    print('All ok')
