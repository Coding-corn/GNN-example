import datetime
import time

import numpy as np
import torch
from matplotlib import animation

np.random.seed(0)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# TODO Need to update values, and revert
plt.rcParams['figure.dpi'] = 96
plt.rcParams.update({'font.size': 24})
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

"""
This script is based on the example provided in https://mlabonne.github.io/blog/posts/2022-03-09-Graph_Attention_Network.html
"""

if __name__ == '__main__':
    # TODO Revert
    epoch = 10
    # epoch = 200
    # Defines the number of epochs between every animation frame
    epochGran = 5

    tic = time.time()

    """Import and analyse dataset"""
    # Import dataset from PyTorch Geometric
    dataset = Planetoid(root=".", name="CiteSeer")
    data = dataset[0]

    # Get the list of degrees for each node
    degrees = degree(data.edge_index[0]).numpy()
    # Count the number of nodes for each degree
    numbers = Counter(degrees)
    # Convert counter to dictionary
    numbers = {int(k): v for k, v in numbers.items()}

    # Bar plot
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Number of Nodes')
    plt.bar(numbers.keys(), numbers.values(), color='#0A047A')
    plt.title("Node Degree Distribution")
    plt.xlim(min(numbers.keys()), max(numbers.keys()))
    plt.savefig('nodeDegree.png', bbox_inches="tight")
    plt.show()

    # TODO Use tSNE visualisation on original dataset
    # TODO Use TSNE visualisation here after training and compare with results before training.
    #  Check perplexity value which garners the lowest KL divergence

    """Create models"""


    class GCN(torch.nn.Module):
        """Graph Convolutional Network"""

        def __init__(self, dim_in, dim_h, dim_out):
            super().__init__()
            self.gcn1 = GCNConv(dim_in, dim_h)
            self.gcn2 = GCNConv(dim_h, dim_out)
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=0.01,
                                              weight_decay=5e-4)

        def forward(self, x, edge_index):
            h = F.dropout(x, p=0.5, training=self.training)
            h = self.gcn1(h, edge_index).relu()
            h = F.dropout(h, p=0.5, training=self.training)
            h = self.gcn2(h, edge_index)
            return h, F.log_softmax(h, dim=1)


    class GAT(torch.nn.Module):
        """Graph Attention Network"""

        def __init__(self, dim_in, dim_h, dim_out, heads=8):
            super().__init__()
            self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
            self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=0.005,
                                              weight_decay=5e-4)

        def forward(self, x, edge_index):
            h = F.dropout(x, p=0.6, training=self.training)
            h = self.gat1(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=0.6, training=self.training)
            h = self.gat2(h, edge_index)
            return h, F.log_softmax(h, dim=1)


    def accuracy(pred_y, y):
        """Calculate accuracy."""
        return ((pred_y == y).sum() / len(y)).item()


    def train(model, data):
        """Train a GNN model and return the trained model."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = model.optimizer

        # Data for animations
        embeddings = []
        outputs = []
        losses = []
        accuracies = []

        model.train()
        for epoch_ in range(epoch + 1):
            # Training
            optimizer.zero_grad()
            h, out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Store data for animations
            if epoch_ % epochGran == 0:
                embeddings.append(h)
                outputs.append(out.argmax(dim=1))
                losses.append(loss)
                accuracies.append(acc)

            # Validation
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

            # Print metrics
            if epoch_ % epochGran == 0:
                print(f'Epoch {epoch_:>3} | Train Loss: {loss:.3f} | Train Acc: '
                      f'{acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc * 100:.2f}%')

        return model, embeddings, outputs, losses, accuracies


    @torch.no_grad()
    def test(model, data):
        """Evaluate the model on the test set and print the accuracy score."""
        model.eval()
        _, out = model(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


    """Train and test models"""
    # TODO Revert to True
    gcnSim = False
    if gcnSim:
        # Create GCN model
        gcn = GCN(dataset.num_features, 16, dataset.num_classes)
        print(gcn)
        # Train and test
        gcn = train(gcn, data)[0]
        acc = test(gcn, data)
        print(f'\nGCN test accuracy: {acc * 100:.2f}%\n')

    # Create GAT model
    gat = GAT(dataset.num_features, 8, dataset.num_classes)
    print(gat)
    # Train and test
    gat, embeddings, outputs, losses, accuracies = train(gat, data)
    acc = test(gat, data)
    print(f'\nGAT test accuracy: {acc * 100:.2f}%\n')

    """Visualise embeddings of untrained GAT"""
    h = embeddings[0]
    # Train TSNE
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())

    # Plot TSNE
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
    plt.title('Embedding at Epoch 0')
    plt.savefig('embedEpoch0.png')
    plt.show()

    """Visualise embeddings of trained GAT"""
    h = embeddings[-1]
    # Train TSNE
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())

    # Plot TSNE
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
    plt.title('Embedding at Epoch ' + str(epoch))
    plt.savefig('embedEpoch' + str(epoch) + '.png')
    plt.show()

    # TODO Revert to True
    animEmbedtSNESave = False
    if animEmbedtSNESave:
        # Create and save animation of tSNE evolution wrt time
        def animEmbedtSNE(i):
            plt.cla()  # Clear the current axes
            h = embeddings[i]
            # Train TSNE
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())
            plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
            plt.title(f'Embedding tSNE\nEpoch {i*epochGran} | Loss: {losses[i]:.2f} | Acc: {accuracies[i] * 100:.2f}%',
                      fontsize=18, pad=20)

        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        anim = animation.FuncAnimation(fig, animEmbedtSNE, frames=np.arange(0, len(embeddings)), interval=500)
        animName = "tSNEEmbedAnim"
        anim.save(animName + ".mp4", writer="ffmpeg")
        anim.save(animName + ".gif", writer="pillow")

    """Plot bar chart of accuracy wrt node degree of trained GAT"""
    # Get model's classifications
    out = outputs[-1]
    # Calculate the degree of each node
    degrees = degree(data.edge_index[0]).numpy()

    # Store accuracy scores and sample sizes
    accuracies = []
    sizes = []

    # Accuracy for degrees between 0 and 5
    for i in range(0, 6):
        mask = np.where(degrees == i)[0]
        accuracies.append(accuracy(out[mask], data.y[mask]))
        sizes.append(len(mask))
    # Accuracy for degrees > 5
    mask = np.where(degrees > 5)[0]
    accuracies.append(accuracy(out[mask], data.y[mask]))
    sizes.append(len(mask))

    # Bar plot
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Accuracy Score')
    ax.set_facecolor('#EFEEEA')
    plt.bar(['0', '1', '2', '3', '4', '5', '>5'],
            accuracies,
            color='#0A047A')
    for i in range(0, 7):
        plt.text(i, accuracies[i], f'{accuracies[i] * 100:.2f}%', ha='center', color='#0A047A')
        plt.text(i, accuracies[i] // 2, sizes[i], ha='center', color='white')
    plt.title('Accuracy Distribution after Training')
    plt.savefig('accNodeDegreeEpoch' + str(epoch) + '.png')
    plt.show()

    # TODO Create and save animation of accuracy
    animNodeAccSave = True
    if animNodeAccSave:
        pass

    toc = time.time()
    print("All simulations completed. Program terminating. Total time taken was",
          str(datetime.timedelta(seconds=toc - tic)))