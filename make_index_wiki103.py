from pathlib import Path
from statistics import mode
from retro_pytorch.retrieval import get_tokenizer, get_bert, tokenize, range_chunked, bert_embed, index_embeddings, reset_folder_, faiss_read_index
from retro_pytorch.retrieval import TMP_PATH, INDEX_FOLDER_PATH

import os
from tqdm import tqdm

from fairseq.data import TokenBlockDataset, Dictionary, MonolingualDataset, data_utils
from retro_pytorch.utils import memmap

import numpy as np
import torch

import faiss
from autofaiss import build_index

def memmap_file_to_chunks_(
    memmap_path,
    *,
    folder,
    shape,
    dtype,
    max_rows_per_file = 500
):
    rows, _ = shape

    with memmap(memmap_path, shape = shape, dtype = dtype, mode = 'r') as f:
        root_path = folder
        reset_folder_(root_path)

        for ind, dim_slice in enumerate(range_chunked(rows, batch_size = max_rows_per_file)):
            filename = root_path / f'{ind:05d}.npy'
            data_slice = f[dim_slice]

            np.save(str(filename), f[dim_slice])
            print(f'saved {str(filename)}')

def index_embeddings(
    embeddings_folder,
    *,
    index_folder = INDEX_FOLDER_PATH,
    index_file = 'knn.index',
    index_infos_file = 'index_infos.json',
    max_index_memory_usage = '32G',
    current_memory_available = '40G'
):
    embeddings_path = embeddings_folder
    index_path = index_folder / index_file

    reset_folder_(index_folder)

    build_index(
        embeddings = str(embeddings_path),
        index_path = str(index_path),
        index_infos_path = str(index_folder / index_infos_file),
        metric_type = "l2",
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available,
        make_direct_map = True,
        should_be_memory_mappable = False,
        use_gpu = torch.cuda.is_available(),
    )

    index = faiss_read_index(index_path)
    return index


BERT_MODEL_DIM = 768

SPLIT_NUM_CHUNKS = {'train': 1612922, 'valid': 3400, 'test': 3837}
CHUNK_SIZE = 64

BSZ = 64

dictionary = Dictionary.load(r'/scratch/gpfs/eonal/nlp/retro/wiki103-data-bin/dict.txt')

def chunks_to_emb(split, offline = False):
    NUM_CHUNKS = SPLIT_NUM_CHUNKS[split]
    emb_shape = (NUM_CHUNKS, BERT_MODEL_DIM)
    chunk_shape = (NUM_CHUNKS, CHUNK_SIZE)
    if not os.path.exists(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/%s.chunks.emb.npy'%(split)):
        with memmap(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/%s.chunks.npy'%(split), shape = chunk_shape, dtype = np.int32, mode = 'r') as f\
            , memmap(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/%s.chunks.emb.npy'%(split), shape = emb_shape, dtype = np.float32, mode = 'w+') as embeddings:

            for dim_slice in range_chunked(NUM_CHUNKS, batch_size=BSZ):
                chunks = f[dim_slice]
                chunk_strs = [dictionary.string(i) for i in chunks]

                ids = tokenize(chunk_strs, offline = offline)
                # print(ids, ids.shape)

                batch_embed = bert_embed(ids, offline = offline)
                # print(batch_embed.shape)

                embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
                print(f'embedded {dim_slice.stop} / {NUM_CHUNKS}')

def make_knn_index():
    # only use train split

    EMBEDDING_FOLDER = Path(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/embeddings')
    INDEX_FOLDER = Path(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/index')

    emb_shape = (SPLIT_NUM_CHUNKS['train'], BERT_MODEL_DIM)

    if not (EMBEDDING_FOLDER / '00000.npy').exists():
        memmap_file_to_chunks_(
            r'/scratch/gpfs/eonal/nlp/experiments/wiki103/train.chunks.emb.npy',
            shape = emb_shape,
            dtype = np.float32,
            folder = EMBEDDING_FOLDER,
            max_rows_per_file = 500,
        )

    if not (INDEX_FOLDER / 'knn.index').exists():
        index = index_embeddings(
            embeddings_folder = EMBEDDING_FOLDER,
            index_folder = INDEX_FOLDER,
        )

    index = faiss_read_index(INDEX_FOLDER / 'knn.index')
    return index

# num_nearest_neighbors = 2
# num_extra_neighbors = 20
# total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1

def retrieve_chunks(index, split, diff_doc=True, num_nearest_neighbors=2, num_extra_neighbors=20):
    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1
    NUM_CHUNKS = SPLIT_NUM_CHUNKS[split]
    emb_shape = (NUM_CHUNKS, BERT_MODEL_DIM)

    with memmap(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/%s.chunks.emb.npy'%(split), shape = emb_shape, dtype = np.float32, mode = 'r') as embeddings\
        , memmap(r'/scratch/gpfs/eonal/nlp/experiments/wiki103/%s.knns.npy'%(split), shape = (NUM_CHUNKS, num_nearest_neighbors), dtype = np.int32, mode = 'w+') as knns:

        for dim_slice in range_chunked(NUM_CHUNKS, batch_size = 500):
            query_vector = embeddings[dim_slice]

            distances, indices = index.search(query_vector, k = total_neighbors_to_fetch)

            # remove self from distances and indices

            query_doc_ids = indices[:, 0]
            distances = distances[:, 1:]
            indices = indices[:, 1:]

            if diff_doc:
                # mask out any neighbors that belong to the same document to -1
                neighbor_from_same_doc = np.logical_and(query_doc_ids[..., None] - 20 <= indices, query_doc_ids[..., None] + 20 >= indices) 

                indices = np.where(neighbor_from_same_doc, -1, indices)
                distances = np.where(neighbor_from_same_doc, 1e3, distances)

                # re-sort indices by updated distances
                indices = np.take_along_axis(indices, np.argsort(distances, axis = 1), axis = 1)

            # store nearest neighbors to knn memmap
            knns[dim_slice] = indices[:, :num_nearest_neighbors]

            print(f'knns calculated for {dim_slice.stop} / {NUM_CHUNKS}')

if __name__=="__main__":
    chunks_to_emb('train', offline = True)
    index = make_knn_index()
    
    retrieve_chunks(index, 'train', diff_doc=True)

    # chunks_to_emb('valid')
    # retrieve_chunks(index, 'valid', diff_doc=False)

    chunks_to_emb('test', offline = True)
    retrieve_chunks(index, 'test', diff_doc=False)
