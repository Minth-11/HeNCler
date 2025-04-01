from ebc import EBC
import os.path
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
from matrix import SparseMatrix

# krijg ijle matrix binnen

# uit BCOT
def read_dataset(dataset, sparse=False):
    data = sio.loadmat(os.path.join('/volume1/scratch/zopdebee/GitHub/HeNCler/BCOT-main/data/', f'{dataset}.mat' ) )
    features = data['fea'].astype(float)
    if not sp.issparse(features):
      features = sp.csr_matrix(features)
    labels = data['gnd'].reshape(-1) - 1
    n_classes = len(np.unique(labels))
    return features, labels, n_classes

features, labels, n_classes = read_dataset('wiki', sparse=True)
print(n_classes)

# converteer naar EBC-formaat (?)






# draai EBC
# ebc = EBC(matrix, [30, 125], 10, 1e-10, 0.01)


# metrieken
