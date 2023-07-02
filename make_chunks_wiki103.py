from pathlib import Path
from retro_pytorch.retrieval import text_folder_to_chunks_

import os
from tqdm import tqdm

from fairseq.data import TokenBlockDataset, Dictionary, MonolingualDataset, data_utils
from retro_pytorch.utils import memmap

import numpy as np

CHUNK_SIZE = 64

def load_data(split, dictionary):
    path = os.path.join(r'/scratch/gpfs/eonal/nlp/retro/wiki103-data-bin', split)
    dataset = data_utils.load_indexed_dataset(path, dictionary)
    # print('loaded {} examples from: {}'.format(len(dataset), path))

    dataset = TokenBlockDataset(dataset, dataset.sizes, CHUNK_SIZE, pad=dictionary.pad(), eos=dictionary.eos(), break_mode="none", include_targets=True)

    dataset = MonolingualDataset(
        dataset, 
        dataset.sizes,
        dictionary,
        dictionary,
        add_eos_for_other_targets=False,
        shuffle=False,
        targets=["future"],
        add_bos_token=False,
    )

    return dataset

def make_chunks(split):
    print('Dealing with split', split)
    dictionary = Dictionary.load(r'/scratch/gpfs/eonal/nlp/retro/wiki103-data-bin/dict.txt')
    dataset = load_data(split, dictionary)
    print(len(dataset))

    tot_chunks = len(dataset)
    if len(dataset[-1]['source']) != CHUNK_SIZE:
        tot_chunks -= 1
    print(tot_chunks)

    with memmap(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/%s.chunks.npy'%(split), shape = (tot_chunks, CHUNK_SIZE), dtype = np.int32, mode = 'w+') as chunks_memmap:
        for i in tqdm(range(tot_chunks)):
            chunks_memmap[i] = dataset[i]['source']
        print(chunks_memmap.shape)

make_chunks('test')
