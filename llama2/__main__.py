import argparse
from typing import List

import torch

from llama2.inference import LLaMA


def parse_args() -> argparse.Namespace:
    """
    This function processes command line arguments.
    """
    parser = argparse.ArgumentParser(description='LLaMA 2 from scratch')

    parser.add_argument('-m', '--mode', default='i', const='i', nargs='?',
                        choices=['i', 'b'],
                        help='interactive or batch mode, deafault: interactive')

    return parser.parse_args()


def batch_inference(prompts: List[str], device: str) -> None:
    # Prepare model
    model = LLaMA.build(
        checkpoints_dir='weights/llama-2-7b/',
        tokenizer_path='weights/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)

def interactive_inference(device: str) -> None:
    # Prepare model
    model = LLaMA.build(
        checkpoints_dir='weights/llama-2-7b/',
        tokenizer_path='weights/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device
    )

    print('Input:')

    while True:
        try:
            user_input = input()
            if user_input == 'q':
                break
            for token in model.text_completion_stream(user_input, max_gen_len=256):
                print(token, end='', flush=True)
            print('\n')
        except KeyboardInterrupt:
            break



def main() -> None:
    # Parse commandline arguments
    args = parse_args()

    torch.manual_seed(0)

    # Select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif  torch.backends.mps.is_available():
        device = 'mps'
    device = 'cpu'
    print(f'Using device: {device}')

    if args.mode == 'i':
        interactive_inference(device)
    elif args.mode == 'b':
        prompts = [
            "Simply put, the theory of relativity states that ",
            "If Google was an Italian company founded in Milan, it would",
            # Few shot promt
            """Translate English to French:

            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>""",
            # Zero shot prompt
            """Tell me if the following person is actually Doraemon disguised as human:
            Name: Umar Jamil
            Decision: 
            """
        ]
        batch_inference(prompts, device)
    else:
        raise ValueError(f'Unknown mode {args.mode}')


if __name__ == '__main__':
    main()
