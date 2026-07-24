print('importing libraries')

import phate
import numpy as np
import sys

file_name = sys.argv[1] + ".npy"

print('reading in data')
node_features = np.load(file_name)

print('applying phate')
reducer = phate.PHATE(n_components=3, knn=15, random_state=42)
embeddings_auto = reducer.fit_transform(node_features)

file_name_out = "res_" + sys.argv[1] + ".npy"

print('saving file')
np.save(file_name_out, embeddings_auto)


print(f'"\nfile saved as {file_name_out}"')
