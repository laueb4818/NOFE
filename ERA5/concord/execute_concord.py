import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
# This script runs headless as a subprocess of the notebook, which exports
# MPLBACKEND=module://matplotlib_inline.backend_inline via %matplotlib inline.
# This env has no matplotlib_inline, so any library that imports matplotlib
# (scanpy, via concord) aborts with "Key backend: ... is not a valid value".
# The runner never plots, so force a real backend instead of inheriting one.
os.environ['MPLBACKEND'] = 'Agg'

print('importing libraries')

import sys
import numpy as np
import torch
import anndata as ad
import concord as ccd

file_name = sys.argv[1] + ".npy"

print('reading in data')
node_features = np.load(file_name)

print('applying concord')
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

adata = ad.AnnData(X=node_features.astype(np.float32))
cur_ccd = ccd.Concord(
    adata=adata,
    input_feature=list(adata.var_names),
    device=device,
    latent_dim=3,
    verbose=False,
    # preload_dense=True: concord-sc has a bug where the num_workers kwarg is
    # accepted but never forwarded to its internal DataLoaderManager, so passing
    # num_workers=0 alone does NOT stop it from spawning ~4 worker subprocesses per
    # DataLoader. preload_dense=True hits a different code path in
    # concord/model/dataloader.py that correctly forces num_workers to 0. Without
    # this, every worker spawn re-imports torch/anndata/concord from scratch (this
    # script runs as a plain .py file, so macOS's 'spawn' start method re-executes
    # it as __main__ in each worker), which made a single date take hours instead
    # of seconds.
    preload_dense=True,
)
cur_ccd.fit_transform(output_key='Concord')
embeddings_auto = adata.obsm['Concord']

file_name_out = "res_" + sys.argv[1] + ".npy"
np.save(file_name_out, embeddings_auto)
print(f'"\nfile saved as {file_name_out}"')
