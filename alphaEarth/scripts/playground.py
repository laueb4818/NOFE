#%%

import sys
import os
import json
import torch
import pickle
import lmdb


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/alphaEarth/scripts/" )
else:
    device = torch.device("cpu")
print(f'using Device: {device}')
sys.path.append("../../src/dataset_utilities.py")


from utilities_alphaEarth import (
    AlphaEarth_Graph_Generator,
    GraphLMDBWriter
)



class GraphLMDBReader:
    """
    Reader for loading graph data from LMDB databases.
    
    Usage:
        reader = GraphLMDBReader('datasets/train.lmdb')
        sample = reader[0]  # Load first sample
    """
    
    def __init__(self, db_path):
        self.env = lmdb.open(db_path, readonly=True, lock=False, map_size=10 * 1024**3)
        with self.env.begin() as txn:
            self.length = int(txn.get(b'__len__').decode())
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            byte_data = txn.get(str(idx).encode())
            return pickle.loads(byte_data)

# %%
reader=GraphLMDBReader('../datasets/subgraph/example_train.lmdb')
graph=reader[0]

# %%
graph.input_affinities
# %%
graph.sigmas

# %%
