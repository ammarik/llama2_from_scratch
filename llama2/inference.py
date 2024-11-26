import argparse
import json
import time
from pathlib import Path
from typing import Generator, Optional

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from llama2.model import ModelArgs, Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs,  device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        self.device = device
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, 'No checkpoint files found!'
            chk_path = checkpoints[0]
            print(f'Loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path, map_location=device)
            print(f'Loaded checkpoint in {time.time() - prev_time:.2f}s')

        with open(Path(checkpoints_dir) / 'params.json', 'r') as f:
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

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time() - prev_time:.2f}s')
        
        return LLaMA(model, tokenizer, model_args, device)

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        # Redistribute probs, since now when we remove low probable tokens, it doesn't sum to 1 anymore.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1) # Sample tokens accordingly to probs - select one.
        next_token = torch.gather(probs_idx, -1, next_token) # We sorted the probs tensor - we need to get original index of the token, so it matches the dictionary
        return next_token


    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not loarger than the maximum seq len
        assert max_prompt_len <= self.args.max_seq_len
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.device) # Create a new tensor of size batch_size x total_len filled by padding tokens
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        
        eos_reached = torch.tensor([False] *  batch_size, device=self.device)
        prompt_tokens_mask = tokens != pad_id # True if token is prompt, False if otherwise

        for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos) # We pass in only one token, and we tell the model, what is the current pos (because of the KV cache)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # If we didn't specified any temperature we just use the greedy strategy
                # Greedily select the token with the maximum probability
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)

            # Only replace the token if it is padding token
            #   In the beginning we already have some tokens that come from the prompt. We need to give
            #   the prompt to the model to build the initial cache - but we don't care about the output 
            #   for these.
            #   So first, we will give in the prompt tokens to the model not because we care about what
            #   the model will output for those tokens but only because we want the KV cache to be built
            #   for those positions. And we only care about what is the model outputting after we give
            #   the last token of the prompt to it.
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id()) # If all of the inputs/outputs in the batch reached EOS - we stop the for loop
            if all(eos_reached):
                break
        
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            print(f'current_prompt_tokens: {current_prompt_tokens}')
            for token in current_prompt_tokens:
                print(f'decoded_token: |{self.tokenizer.decode(token)}|')
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)


    def text_completion_stream(self, prompt: str, temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None) -> Generator:
        # Convert prompt into tokens
        prompt_tokens = self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
        #print(f'Prompt tokens: {prompt_tokens}')
        
        # Batch size = 1
        batch_size = 1
        assert batch_size <= self.args.max_batch_size

        # Prompt length & Make sure the prompt length is not longer than the maximum seq len
        prompt_len = len(prompt)
        assert prompt_len <= self.args.max_seq_len
        #print(f'Prompt len: {prompt_len}')

        # Max total length
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        max_total_len = min(self.args.max_seq_len, max_gen_len + prompt_len)
        #print(f'Max total len: {max_total_len}')
        
        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((max_total_len,), pad_id, dtype=torch.long, device=self.device) # Create a new tensor of size batch_size x total_len filled by padding tokens
        print(f'Tokens: {tokens}')
        tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)
        print(f'Tokens: {tokens}')

        # prompt_tokens_mask = tokens != pad_id # True if token is prompt, False if otherwise
        # print(f'prompt_tokens_mask: {prompt_tokens_mask}')

        out_tokens = []
        decoded_tokens = []

        for cur_pos in range(1, max_total_len):
            with torch.no_grad():
                extended_tokens = tokens[None, :] # Introduce one extra dimension, since model expects batch dimenion.
                logits = self.model.forward(extended_tokens[:, cur_pos-1:cur_pos], cur_pos) # We pass in only one token, and we tell the model, what is the current pos (because of the KV cache)
                #print(f'Logits: {logits}')
                #print(f'Logits: {logits[0, 0, :]}')
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[0, 0, :] / temperature, dim=-1)
                #print(f'Probs: {probs}')
                next_token = self._sample_top_p(probs, top_p)
            else:
                # If we didn't specified any temperature we just use the greedy strategy
                # Greedily select the token with the maximum probability
                next_token = torch.argmax(logits[0, 0, :], dim=-1)
            
            # Only replace the token if it is padding token
            #   In the beginning we already have some tokens that come from the prompt. We need to give
            #   the prompt to the model to build the initial cache - but we don't care about the output 
            #   for these.
            #   So first, we will give in the prompt tokens to the model not because we care about what
            #   the model will output for those tokens but only because we want the KV cache to be built
            #   for those positions. And we only care about what is the model outputting after we give
            #   the last token of the prompt to it.
            #print(f'next_token: {next_token}')
            #print(f'tokens[cur_pos]: {tokens[cur_pos]}')
            #print(f'pad id: {pad_id}')

            if tokens[cur_pos] == pad_id:
                if (next_token == self.tokenizer.eos_id()):
                    # EOS is reached only if we found an EOS token for padding position
                    break

                tokens[cur_pos] = next_token
                # Return current token
                #print(f'next_token[0]: {next_token[0]}')
                #print(f'next_token.tolist(): {next_token.tolist()}')
                #print(f'next_token: {next_token}')

                prev_len = len(decoded_tokens)
                out_tokens = out_tokens + next_token.tolist()

                #print(f'\n\nOUT TOKENS |{out_tokens}|\n\n')
                decoded_tokens = self.tokenizer.decode(out_tokens)
                #print(f'decoded_tokens: |{decoded_tokens}|')

                decoded_token = decoded_tokens[(prev_len-len(decoded_tokens)):]
                
                
                #print(f'decoded_token: &{decoded_token}&')
                yield decoded_token




# Tokens:  P,P,P,n,_,_,_