import argparse
import os.path
import sys
from datasets import load_from_disk
import pandas as pd
from transformers import AutoTokenizer, pipeline
from model_loader import *
from evaluate import load
import regex as re
import numpy as np
from nltk import word_tokenize
from tqdm import tqdm


def remove_dup_lines(text):
    arr = text.split('\n')
    new_arr = []
    for x in arr:
        x = x.strip()
        if x not in new_arr and len(x) > 0:
            new_arr.append(x)
    return '\n'.join(new_arr)


def load_data(args):
    data_type = args.model_type if 'frost' not in args.model_type else 'plan'
    base_dir = '/nlp/projects/summarization/bhc_data_cleanup/llama_inference/'
    data_dir = os.path.join(base_dir, f'{args.dataset}_{data_type}_{args.prompt_window}')
    print(f'Reading in data from {data_dir}')
    data = load_from_disk(data_dir)
    return data


def main(args, data):
    tokenizer_model = 'NousResearch/Llama-2-7b-hf'  # Switched from args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, model_max_length=sys.maxsize, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token

    if args.ckpt == 'final':
        model_path = args.model
    else:
        # Use latest checkpoint
        ckpts = [x for x in os.listdir(args.model) if 'step' in x]
        print(f'Checkpoints:\n', '\n'.join(ckpts))
        if args.ckpt == 'latest':
            steps = [int(x.split('_')[1]) for x in ckpts]
            ckpt_idx = int(np.argmax(steps))
        else:
            ckpt_idx = int([x.split('_')[1] for x in ckpts].index(args.ckpt))

        model_path = os.path.join(args.model, ckpts[ckpt_idx])
    print(f'Loading from {model_path}...')
    model = load_model_and_apply_patches(model_path, args)

    pipe = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,  # pad_token_id=tokenizer.eos_token_id,
        # repetition_penalty=args.repetition_penalty,
        temperature=args.temperature, do_sample=args.temperature > 0,
        #  penalty_alpha=args.penalty_alpha, top_k=args.top_k,
    )

    outputs = []

    for row in tqdm(data, total=len(data)):
        prompt = row['prompt']

        adj_max_new_tokens = args.max_new_tokens
        # Remove the plan -- we are generating it
        if 'frost' in args.model_type and args.frost_strategy == 'self':
            truncate_idx = re.search(r'### ENTITIES', prompt).end() + 1
            prompt = prompt[:truncate_idx]
            adj_max_new_tokens += 256  # Make room to generate a plan

        n = len(prompt)
        response = pipe(
            prompt, num_return_sequences=1, max_new_tokens=adj_max_new_tokens
        )[0]["generated_text"][n:]

        if 'frost' in args.model_type and args.frost_strategy == 'self':
            if '### BRIEF HOSPITAL COURSE:' in response:
                response = [x.strip() for x in response.split('### BRIEF HOSPITAL COURSE:') if len(x.strip()) > 0][-1]
            else:
                print('Didnt generate a summary. Re-prompting.')
                prompt += response + '\n\n' + '### BRIEF HOSPITAL COURSE:\n'
                n = len(prompt)
                response = pipe(
                    prompt, num_return_sequences=1, max_new_tokens=args.max_new_tokens
                )[0]["generated_text"][n:]

        response = remove_dup_lines(response)
        num_toks = len(word_tokenize(response))

        obj = rouge.compute(
            references=[row['reference']], predictions=[response], use_aggregator=False
        )
        r1 = obj['rouge1'][0]
        r2 = obj['rouge2'][0]

        outputs.append({
            'example_id': row['example_id'],
            'prediction': response,
            'reference': row['reference'],
            'target_sents': '\n'.join(row['target_sents']),
            'rouge1': r1,
            'rouge2': r2,
            'pred_tokens': num_toks,
        })

        avg_r1 = np.mean([x['rouge1'] for x in outputs])
        avg_r2 = np.mean([x['rouge2'] for x in outputs])
        avg_tokens = np.mean([x['pred_tokens'] for x in outputs])
        print(f'\n\n{response}\n\n')
        print(f'R1: {round(avg_r1, 2)}. R2: {round(avg_r2, 2)}. Tokens: {round(avg_tokens, 2)}')
    return pd.DataFrame(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 'NousResearch/Llama-2-7b-hf'
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--prompt-window', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.0)
    # parser.add_argument('--repetition-penalty', type=float, default=1.25)
    # parser.add_argument('--penalty-alpha', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--ckpt', default='latest')

    parser.add_argument('--model-type', default='baseline', choices=[
        'frost', 'frost_plan', 'baseline', 'decorate', 'clique_frost'
    ])
    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--frost-strategy', default='follow', choices=['self', 'follow'])

    args = add_args(parser).parse_args()

    if args.frost_strategy == 'self':
        assert args.model_type == 'frost_plan'

    rouge = load('rouge', keep_in_memory=True)

    suffix = '-none' if args.model_type == 'baseline' else f'-{args.model_type}'
    args.model = os.path.expanduser(f'/nlp/projects/summarization/bhc_data_cleanup/bhc_weights/yarn-7b-8k{suffix}')
    print(f'Loading model from {args.model}')
    out_fn = os.path.join(args.model, f'{args.dataset}_{args.ckpt}_{args.prompt_window}')
    if 'frost' in args.model_type:
        out_fn += f'_{args.frost_strategy}'

    out_fn += '_results.csv'
    args.yarn = 16.0
    args.finetuned = True
    args.original_max_position_embeddings = 4096
    args.flash_attention = True
    args.custom_model_together = True

    data = load_data(args)
    predictions = main(args, data)

    print(f'Saving to {out_fn}')
    predictions.to_csv(out_fn, index=False)
