from random import shuffle
import torch
from retro_pytorch import RETRO, TrainingWrapper
from retro_pytorch.utils import memmap
from pathlib import Path

from tqdm import tqdm
import time
import os
import numpy as np

import random
import time
import math
import glob

from fairseq.data import TokenBlockDataset, Dictionary, MonolingualDataset, data_utils
from fairseq.optim.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from fairseq.optim.adam import FairseqAdam
from torch.utils.data import DataLoader
import argparse
import wandb

DATA_PATH = Path('/scratch/gpfs/eonal/nlp/retro/wiki103-data-bin')
CHUNKS_PATH = Path('/scratch/gpfs/eonal/nlp/experiments/wiki103')
PROJECT_NAME = 'RETRO-wiki103'
WANDB_MODE = 'offline'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(1)

def load_data(split, dictionary):
    path = os.path.join(DATA_PATH, split)
    dataset = data_utils.load_indexed_dataset(path, dictionary)
    # print('loaded {} examples from: {}'.format(len(dataset), path))

    dataset = TokenBlockDataset(dataset, dataset.sizes, 512, pad=dictionary.pad(), eos=dictionary.eos(), break_mode="none", include_targets=True)

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

def load_last_checkpoint(init_retro, save_dir):
    ckps = glob.glob('%s/epc*-*.ckp'%(save_dir))
    if len(ckps) == 0:
        return init_retro, 0, 0

    epoch = 0
    num_updates = 0

    for ckp in ckps:
        basename = os.path.basename(ckp)[3:-4]
        ckp_epoch = int(basename.split('-')[0])
        ckp_update = int(basename.split('-')[1])
        if ckp_update > num_updates:
            epoch = ckp_epoch
            num_updates = ckp_update
            retro = torch.load(ckp)

    print('Loaded the checkpoint at step %d, epoch %d'%(num_updates, epoch))
    return retro, epoch + 1, num_updates


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0005, type=float, metavar='LR')
parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                    help='warmup the learning rate linearly for the first N updates')
parser.add_argument('--warmup-init-lr', default=1e-07, type=float, metavar='LR',
                    help='initial learning rate during warmup phase; default is args.lr')
parser.add_argument('--adam-betas', default='(0.9, 0.98)', metavar='B',
                            help='betas for Adam optimizer')
parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                    help='epsilon for Adam optimizer')
parser.add_argument('--weight-decay', '--wd', default=0.1, type=float, metavar='WD',
                    help='weight decay')
parser.add_argument('--fp16-adam-stats', default=False, type=bool)
parser.add_argument('--disable-retro', default=False, action='store_true')
parser.add_argument('--save-dir', default=None, type=str)
parser.add_argument('--no-deepnet', default=False, action="store_true")
parser.add_argument('--mixed-precision', action="store_true")
parser.add_argument('--bm25', action="store_true")
args = parser.parse_args()
print(args)

# Process data
dictionary = Dictionary.load(os.path.join(DATA_PATH, 'dict.txt'))
train_data = load_data('train', dictionary)
valid_data = load_data('valid', dictionary)

print(len(valid_data))
print(len(train_data))

SEQ_LEN = 512
CHUNK_SIZE = 64
T = 500
K = 2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    use_deepnet = not(args.no_deepnet),       # turn on post-normalization with DeepNet residual scaling and initialization, for scaling to 1000 layers
    share_input_output_embed = True,
    pad_id=dictionary.pad(),
)
# print(retro)

print('RETRO', count_parameters(retro))
print('encoder', count_parameters(retro.encoder))
print('decoder', count_parameters(retro.decoder))
print('RETRO encoder only', count_parameters(retro) - count_parameters(retro.encoder))

EFFECTIVE_BSZ=64
BSZ = 16
NUM_EPOCHS = 20
NUM_CHUNKS = 1612922
MAX_UPDATES = 50000
GRAD_ACCU = EFFECTIVE_BSZ // BSZ

train_dataloader = DataLoader(train_data, batch_size=BSZ, shuffle=True, collate_fn=train_data.collater)
valid_dataloader = DataLoader(valid_data, batch_size=BSZ, shuffle=False, collate_fn=train_data.collater)

# Train!

if not os.path.exists(args.save_dir):
    print('Make dir %s'%(args.save_dir))
    os.mkdir(args.save_dir)


if not args.disable_retro:
    if args.bm25:
        train_knns_path = CHUNKS_PATH / 'train.bm25_knns.npy'
        print('Using BM25 Pre-Retrieved KNNs from', train_knns_path)
    else:
        train_knns_path = CHUNKS_PATH / 'train.knns.npy'
        print('Using BERT Pre-Retrieved KNNs from', train_knns_path)
        
    train_knns = np.memmap(train_knns_path, shape = (NUM_CHUNKS, K), dtype = np.int32, mode = 'r')
else:
    print('Retrieval disabled.')

train_chunks = np.memmap(CHUNKS_PATH / 'train.chunks.npy', shape = (NUM_CHUNKS, CHUNK_SIZE), dtype = np.int32, mode = 'r')


wandb.init(project=PROJECT_NAME, mode=WANDB_MODE, config=args, name=os.path.basename(args.save_dir), settings=wandb.Settings(start_method='fork'))


retro, epoch, num_updates = load_last_checkpoint(retro, args.save_dir)

retro.cuda()
retro.train()
wandb.watch(retro)

n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    print('Move model to multi gpus (%d)'%(n_gpu))
    retro = torch.nn.DataParallel(retro)

optimizer = FairseqAdam(args, retro.parameters())
lr_scheduler = InverseSquareRootSchedule(args, optimizer)
lr_scheduler.step_update(num_updates)

if args.mixed_precision:
    scaler = torch.cuda.amp.GradScaler()

while epoch < NUM_EPOCHS and num_updates < MAX_UPDATES:
    acc_loss = 0.0
    tot_token = 0
    grad_acc_cnt = 0
    st = time.time()
    st_num_updates = num_updates
    for i, batch in tqdm(enumerate(train_dataloader)):
        seq = batch['net_input']['src_tokens'].cuda()
        labels = batch['target'].cuda()
        retrieved = None

        if not args.disable_retro:
            ids = batch['id']
            chunk_ids = (ids * (SEQ_LEN // CHUNK_SIZE)).view(-1, 1) + torch.arange(0, SEQ_LEN // CHUNK_SIZE).view(1, -1)

            chunk_ids = np.clip(chunk_ids, 0, NUM_CHUNKS - 1)
            retrieved_ids = train_knns[chunk_ids] # b x n x k
            retrieved = train_chunks[retrieved_ids] # b x n x k x c

            continuation_ids = np.clip(retrieved_ids + 1, 0, NUM_CHUNKS - 1)
            continuation = train_chunks[continuation_ids] # b x n x k x c

            retrieved = torch.tensor(np.concatenate((retrieved, continuation), axis=-1)).cuda()
        else:
            retrieved = None

        with torch.cuda.amp.autocast():
            loss, num_token = retro(seq, retrieved, return_loss=True, labels=labels)
            loss = loss.sum()
            num_token = num_token.sum()

        acc_loss += loss.cpu().item()
        tot_token += num_token.cpu().item()

        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_acc_cnt += 1

        if (grad_acc_cnt >= GRAD_ACCU) or (i == len(train_dataloader) - 1) or (num_updates >= MAX_UPDATES):
            grad_acc_cnt = 0
            num_updates += 1
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step_update(num_updates)
            optimizer.zero_grad()

            used_time = time.time() - st
            wps = tot_token / used_time
            ups = (num_updates - st_num_updates) / used_time
            curr_loss = loss.cpu().item() / num_token.cpu().item() / math.log(2)
            print('Epoch %d, %d/%d(%.1f%%): avg loss %.3f curr loss %.3f (used time: %.1f, ups: %.1f)'%(epoch, i, len(train_dataloader), (i / len(train_dataloader) * 100), acc_loss / tot_token / math.log(2), curr_loss, used_time, ups))

            log_output = {
                'loss': acc_loss / tot_token / math.log(2),
                'ppl': math.exp(acc_loss / tot_token),
                'lr': optimizer.get_lr(),
                'wps': tot_token / used_time,
                'ups': (num_updates - st_num_updates) / used_time,
            }

            wandb.log(log_output, step=num_updates)

            # save model
            if (i == len(train_dataloader) - 1) or (num_updates >= MAX_UPDATES):
                ckp_path = '%s/epc%d-%d.ckp'%(args.save_dir, epoch, num_updates)
                model_to_save = retro.module if hasattr(retro, 'module') else retro
                torch.save(model_to_save, ckp_path)
                print('Saved model checkpoint to %s'%(ckp_path))

        if num_updates >= MAX_UPDATES:
            break

    if num_updates >= MAX_UPDATES:
        break

    epoch += 1
