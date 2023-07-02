### BM25 Index creation
from pathlib import Path
from fairseq.data import Dictionary
import numpy as np
from retro_pytorch.utils import memmap
from retro_pytorch.retrieval import range_chunked
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
import subprocess
import os
import json

DATA_PATH = Path('/scratch/gpfs/eonal/nlp/retro/wiki103-data-bin')
CHUNKS_PATH = Path('/scratch/gpfs/eonal/nlp/experiments/wiki103')

dictionary = Dictionary.load(DATA_PATH / 'dict.txt')
SPLIT_NUM_CHUNKS = {'train': 1612922, 'valid': 3400, 'test': 3837}
CHUNK_SIZE = 64


def memmap_file_to_chunks_json_():
  # get chunks in json format
  print('Creating chunks JSON file...')

  NUM_CHUNKS = SPLIT_NUM_CHUNKS['train']
  chunk_shape = (NUM_CHUNKS, CHUNK_SIZE)

  with memmap(CHUNKS_PATH / 'train.chunks.npy', shape = chunk_shape, dtype = np.int32, mode = 'r') as chunks:
    json_data = [{"id": str(i), "contents":dictionary.string(chunks[i])} for i in tqdm(range(chunk_shape[0]))]

  JSON_PATH = CHUNKS_PATH / 'chunks_json'
  if not os.path.exists(JSON_PATH):  
    os.mkdir(JSON_PATH)

  with open(JSON_PATH / 'chunks.json', 'w') as f:
    json.dump(json_data, f)
  print('Chunks saved in JSON format in ', JSON_PATH / 'chunks.json')


def make_bm25_index(RECREATE_JSON = False, RECREATE_INDEX=False):
  NUM_THREADS = 8
  INDEX_PATH = CHUNKS_PATH / 'index/BM25_index'

  if RECREATE_JSON or not os.path.exists(CHUNKS_PATH / 'chunks_json/chunks.json'):
    memmap_file_to_chunks_json_()

  if RECREATE_INDEX or not os.path.exists(INDEX_PATH):
    print('Building BM25 index...')

    # build index
    subprocess.run(f"python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input {CHUNKS_PATH / 'chunks_json'} \
      --index {INDEX_PATH} \
      --generator DefaultLuceneDocumentGenerator \
      --threads {NUM_THREADS}", shell=True)
    
    print('BM25 index successfully created.')  
  else:
    print(f'Using pre-built BM25 index at {INDEX_PATH}. Use RECREATE_INDEX=True to force recreate the index.')
  
  print('Index statistics:')
  index_reader = IndexReader(INDEX_PATH)
  print(index_reader.stats())

def retrieve_chunks_bm25(split, diff_doc=True, num_nearest_neighbors=2, num_extra_neighbors=20):
  ### BM25 PRE-RETRIEVAL
  NUM_THREADS=8
  INDEX_PATH = CHUNKS_PATH / 'index/BM25_index'
  if not os.path.exists(INDEX_PATH):
    raise Exception(f'Cannot find lucene index at: "{INDEX_PATH}"')

  # set BM25 search parameters
  searcher = LuceneSearcher(INDEX_PATH)
  searcher.set_bm25(k1 = 0.9, b = 0.4)

  total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1

  NUM_CHUNKS = SPLIT_NUM_CHUNKS[split]
  chunk_shape = (NUM_CHUNKS, CHUNK_SIZE)

  with memmap(CHUNKS_PATH / f'{split}.chunks.npy', shape = chunk_shape, dtype = np.int32, mode = 'r') as chunks\
  , memmap(CHUNKS_PATH / f'{split}.bm25_knns.npy', shape = (NUM_CHUNKS, num_nearest_neighbors), dtype = np.int32, mode = 'w+') as knns:
    
    for dim_slice in range_chunked(NUM_CHUNKS, batch_size = 500):
      query_vector = chunks[dim_slice]
      query_strs = [dictionary.string(query) for query in query_vector] # chunk_strs
      query_ids = np.arange(dim_slice.start, dim_slice.stop) # chunk_ids
      
      hits = searcher.batch_search(queries = query_strs, qids = [str(qid) for qid in query_ids], k = total_neighbors_to_fetch, threads=NUM_THREADS)
      search_results = np.array([hits[str(i)] for i in range(dim_slice.start, dim_slice.stop)])

      # remove the query chunk from retrieved neighbor chunk scores/indices
      scores = np.array([list(map(lambda x: x.score, neighbors)) for neighbors in search_results[:, 1:]])
      indices = np.array([list(map(lambda x: int(x.docid), neighbors)) for neighbors in search_results[:, 1:]])

      if diff_doc:
        # mask out neighbors that belong to the same document with -1 idx and -1e3 BM25 score
        neighbor_from_same_doc = np.logical_and(query_ids[..., None] - 20 <= indices, query_ids[..., None] + 20 >= indices)

        indices = np.where(neighbor_from_same_doc, -1, indices)
        scores = np.where(neighbor_from_same_doc, -1e3, scores)

        # re-sort indices by updated scores
        indices = np.take_along_axis(indices, np.argsort(scores, axis = 1)[:, ::-1], axis = 1)

      # store nearest neighbors to knn memmap
      knns[dim_slice] = indices[:, :num_nearest_neighbors]
      
      print(f'knns calculated for {dim_slice.stop} / {NUM_CHUNKS}')

if __name__=="__main__":  
  make_bm25_index(RECREATE_INDEX=False)

  retrieve_chunks_bm25('train', diff_doc=True)
  retrieve_chunks_bm25('test', diff_doc=False)
  