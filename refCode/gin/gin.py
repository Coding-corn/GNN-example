import datetime
import time

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
from packages.training import *
from packages.loader import *
from packages.utils import *

"""
This script is based on the example provided in https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
"""

if __name__ == '__main__':
    epoch = 100
    # Defines the number of epochs between every animation frame
    epoch_gran = 10

    tic = time.time()

    """Load ana analyse dataset."""
    dataset = TUDataset(root='.', name='PROTEINS').shuffle()
    anim_graph(dataset)

    """Create mini-batches for training, validation, and test sets."""
    train_loader, val_loader, test_loader = loader(dataset)

    """Create models"""
    gcn = GCN(dataset, dim_h=32)
    gin = GIN(dataset, dim_h=32)

    """Train, validate, and test models"""
    print("GCN:")
    gcn = train(gcn, train_loader, val_loader, test_loader, epoch, epoch_gran)
    print("GIN:")
    gin = train(gin, train_loader, val_loader, test_loader, epoch, epoch_gran)

    """Visualise graph classification results of trained models"""
    plot_cat_graph(dataset, gcn, 'GCN - Graph classification', 'gcnCatGraph')
    plot_cat_graph(dataset, gin, 'GIN - Graph classification', 'ginCatGraph')

    """Create ensemble model by combining the outputs of both the GCN and GIN models"""
    gcn.eval()
    gin.eval()
    acc_gcn = 0
    acc_gin = 0
    acc = 0

    for data in test_loader:
        # Get classifications
        _, out_gcn = gcn(data.x, data.edge_index, data.batch)
        _, out_gin = gin(data.x, data.edge_index, data.batch)
        out = (out_gcn + out_gin) / 2

        # Calculate accuracy scores
        acc_gcn += accuracy(out_gcn.argmax(dim=1), data.y) / len(test_loader)
        acc_gin += accuracy(out_gin.argmax(dim=1), data.y) / len(test_loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(test_loader)

    # Print results
    print(f'GCN accuracy:     {acc_gcn * 100:.2f}%')
    print(f'GIN accuracy:     {acc_gin * 100:.2f}%')
    print(f'GCN+GIN accuracy: {acc * 100:.2f}%\n')

    toc = time.time()
    print("All simulations completed. Program terminating. Total time taken was",
          str(datetime.timedelta(seconds=toc - tic)))
