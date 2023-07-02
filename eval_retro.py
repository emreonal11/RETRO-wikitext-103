from random import shuffle
import torch
from retro_pytorch import RETRO, TrainingWrapper
from retro_pytorch.utils import memmap

from einops import rearrange

from tqdm import tqdm
import time
import os
import numpy as np

import random
import time
import math
import glob

from fairseq.data import TokenBlockDataset, Dictionary, MonolingualDataset, data_utils, LMContextWindowDataset
from fairseq.optim.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from fairseq.optim.adam import FairseqAdam
from torch.utils.data import DataLoader

import argparse

import wandb

SPLIT_NUM_CHUNKS = {'train': 1612922, 'valid': 3400, 'test': 3837}
SEQ_LEN = 512
CHUNK_SIZE = 64
T = 500
K = 2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(1)

def load_data(split, dictionary, toens_per_sample=512, context_window=0):
    path = os.path.join('/u/zzhong/nlpzzhong/repos/rep-knn/knnlm/data-bin/wikitext-103', split)
    dataset = data_utils.load_indexed_dataset(path, dictionary)
    # print('loaded {} examples from: {}'.format(len(dataset), path))

    dataset = TokenBlockDataset(dataset, dataset.sizes, toens_per_sample - context_window, pad=dictionary.pad(), eos=dictionary.eos(), break_mode="none", include_targets=True)

    dataset = MonolingualDataset(
        dataset, 
        dataset.sizes,
        dictionary,
        dictionary,
        add_eos_for_other_targets=False,
        shuffle=(True if split == "train" else False),
        targets=["future"],
        add_bos_token=False,
    )

    return dataset

def load_model(model_path):
    model = torch.load(model_path)
    return model

def retro_evaluate(retro, dataloader, enable_retrieval=False, chunks=None, knns=None, context_window=0):
    if enable_retrieval:
        assert chunks is not None
        assert knns is not None
        DATASTORE_SIZE = chunks.shape[0]
        NUM_CHUNKS = knns.shape[0]
    
    retro.eval()
    retro.cuda()

    acc_loss = 0.0
    tot_token = 0

    st = time.time()

    for i, batch in tqdm(enumerate(dataloader)):
        seq = batch['net_input']['src_tokens'].cuda()
        labels = batch['target'].cuda()
        retrieved = None

        if enable_retrieval:
            ids = batch['id']

            if 'start_indices' in batch:
                start_indices = torch.tensor(batch['start_indices'])
                chunk_ids = (ids * ((SEQ_LEN - context_window) // CHUNK_SIZE)).view(-1, 1) + torch.arange(0, seq.shape[-1] // CHUNK_SIZE).view(1, -1) \
                            - (start_indices.view(-1, 1) // CHUNK_SIZE)
            else:
                chunk_ids = (ids * ((SEQ_LEN - context_window) // CHUNK_SIZE)).view(-1, 1) + torch.arange(0, seq.shape[-1] // CHUNK_SIZE).view(1, -1)
            
            chunk_ids = np.clip(chunk_ids, 0, NUM_CHUNKS - 1)
            retrieved_ids = knns[chunk_ids] # b x n x k
            retrieved = chunks[retrieved_ids] # b x n x k x c

            continuation_ids = np.clip(retrieved_ids + 1, 0, DATASTORE_SIZE - 1)
            continuation = chunks[continuation_ids] # b x n x k x c

            retrieved = torch.tensor(np.concatenate((retrieved, continuation), axis=-1)).cuda()
        else:
            retrieved = None

        with torch.no_grad():
            loss, num_token = retro(seq, retrieved, return_loss=True, labels=labels)
            loss = loss.sum()
            num_token = num_token.sum()

        acc_loss += loss.cpu().item()
        tot_token += num_token.cpu().item()
    
    runtime = time.time() - st

    eval_result = {
        'loss': acc_loss / tot_token / math.log(2),
        'ppl': math.exp(acc_loss / tot_token),
        'num_token': tot_token, 
        'speed': tot_token / runtime, 
    }

    return eval_result

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='model path to evaluate')
    parser.add_argument('--split', default='valid', type=str, help='which subset to evaluate')
    parser.add_argument('--tokens-per-sample', default=512, type=int)
    parser.add_argument('--context-window', default=0, type=int)
    args = parser.parse_args()
    print(args)

    # Process data
    dictionary = Dictionary.load('/u/zzhong/nlpzzhong/repos/rep-knn/knnlm/data-bin/wikitext-103/dict.txt')
    dataset = load_data(args.split, dictionary, args.tokens_per_sample, args.context_window)

    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=dictionary.pad(),
        )

    retro = RETRO(
        num_tokens = len(dictionary),
        chunk_size = CHUNK_SIZE,                         # the chunk size that is indexed and retrieved (needed for proper relative positions as well as causal chunked cross attention)
        max_seq_len = SEQ_LEN,                      # max sequence length
        enc_dim = 512,                           # encoder model dim
        enc_depth = 2,                           # encoder depth
        dec_dim = 512,                           # decoder model dim
        dec_depth = 6,                          # decoder depth
        dec_cross_attn_layers = (3, 6),   # decoder cross attention layers (with causal chunk cross attention)
        heads = 8,                               # attention heads
        dim_head = 64,                           # dimension per head
        dec_attn_dropout = 0.0,                 # decoder attention dropout
        dec_ff_dropout = 0.1,                   # decoder feedforward dropout
        share_input_output_embed = True,
        pad_id=dictionary.pad(),
    )

    if args.model is not None:
        retro = load_model(args.model)

    print('RETRO', count_parameters(retro))
    print('encoder', count_parameters(retro.encoder))
    print('decoder', count_parameters(retro.decoder))

    BSZ = 8

    eval_dataloader = DataLoader(dataset, batch_size=BSZ, shuffle=False, collate_fn=dataset.collater)
    print('Dataset size', len(eval_dataloader))

    # datastore is built on training set
    datastore_chunks = np.memmap('experiments/wiki103/train.chunks.npy', shape = (SPLIT_NUM_CHUNKS['train'], CHUNK_SIZE), dtype = np.int32, mode = 'r')
    
    # pre-computed knn results
    retrieval_knns = np.memmap('experiments/wiki103/%s.knns.npy'%(args.split), shape = (SPLIT_NUM_CHUNKS[args.split], K), dtype = np.int32, mode = 'r')

    eval_result = retro_evaluate(retro, eval_dataloader, enable_retrieval=True, chunks=datastore_chunks, knns=retrieval_knns, context_window=args.context_window)
    print(eval_result)

    eval_result = retro_evaluate(retro, eval_dataloader, enable_retrieval=False, context_window=args.context_window)
    print(eval_result)

