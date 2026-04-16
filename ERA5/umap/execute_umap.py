print('importing libraries')

import umap
import numpy as np
import sys

file_name = sys.argv[1] + ".npy"

print('reading in data')
node_features = np.load(file_name)

print('applying umap')
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_auto = reducer.fit_transform(node_features)

file_name_out = "res_" + sys.argv[1] + ".npy"

print('saving file')
np.save(file_name_out, embeddings_auto)


print(f'"\nfile saved as {file_name_out}"')
