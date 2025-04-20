import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from matplotlib import animation
import time
import datetime

"""
This script is based on the example provided in https://mlabonne.github.io/blog/posts/2022_02_20_Graph_Convolution_Network.html
"""

if __name__ == '__main__':
    dpi = 96
    epoch = 100
    # Defines the number of epochs between every animation frame
    epochGran = 5

    tic = time.time()

    """Import and analyse dataset"""
    # Import dataset from PyTorch Geometric
    dataset = KarateClub()
    data = dataset[0]
    # Convert the edge indices from coordinate list (COO) format to an adjacency matrix
    A = to_dense_adj(data.edge_index)[0].numpy().astype(int)

    # Visualize the graph with its correct labels
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=0),
                     with_labels=True,
                     node_size=800,
                     node_color=data.y,
                     cmap="hsv",
                     vmin=-2,
                     vmax=3,
                     width=0.8,
                     edge_color="grey",
                     font_size=14
                     )
    plt.title("True Labels")
    plt.savefig('trueLabels.png', dpi=dpi)
    plt.show()

    """Build graph convolutional network"""


    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = GCNConv(dataset.num_features, 3)
            self.out = Linear(3, dataset.num_classes)

        def forward(self, x, edge_index):
            h = self.gcn(x, edge_index).relu()
            z = self.out(h)
            return h, z


    model = GCN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


    # Calculate accuracy
    def accuracy(pred_y, y):
        return (pred_y == y).sum() / len(y)


    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []

    # Training loop
    for epoch_ in range(epoch + 1):
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        h, z = model(data.x, data.edge_index)
        # Calculate loss function
        loss = criterion(z, data.y)
        # Calculate accuracy
        acc = accuracy(z.argmax(dim=1), data.y)
        # Compute gradients
        loss.backward()
        # Tune parameters
        optimizer.step()

        # Store data for animations
        embeddings.append(h)
        losses.append(loss)
        accuracies.append(acc)
        outputs.append(z.argmax(dim=1))

        # Print metrics every epochGran number of epochs
        if epoch_ % epochGran == 0:
            print(f'Epoch {epoch_:>3} | Loss: {loss:.2f} | Acc: {acc * 100:.2f}%')

    """Animate node classification process during training regime"""
    # Animate results
    plt.rcParams["animation.bitrate"] = 3000


    def animateNodeClass(i):
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=True,
                         node_size=800,
                         node_color=outputs[i],
                         cmap="hsv",
                         vmin=-2,
                         vmax=3,
                         width=0.8,
                         edge_color="grey",
                         font_size=14
                         )
        plt.title(f'Node Classification\nEpoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i] * 100:.2f}%',
                  fontsize=18, pad=20)


    # Create figure
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    # Create animation
    anim = animation.FuncAnimation(fig, animateNodeClass, frames=np.arange(0, epoch, epochGran), interval=500,
                                   repeat=True)
    animName = "trainLabelAnim"
    # Save animation to video (optional)
    anim.save(animName + ".mp4", writer="ffmpeg")
    # Or to GIF instead
    anim.save(animName + ".gif", writer="pillow")

    """For 3D visualisation of embeddings at the 0-th epoch"""
    # Get first embedding at epoch = 0
    embed = embeddings[0].detach().cpu().numpy()

    # Create a figure and a 3D subplot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    # Set plot configurations
    ax.set_box_aspect([1, 1, 1])  # Maintain an equal aspect ratio for all axes
    # Scatter plot in 3D
    scatter = ax.scatter(
        embed[:, 0], embed[:, 1], embed[:, 2],  # x, y, z coordinates
        s=200,  # Size of points
        c=data.y,  # Colours based on labels
        cmap="hsv",  # Colour map
        vmin=-2, vmax=3  # Colour limits for consistency
    )
    # Ensure proper display
    ax.set_title('3D Node Embedding at Epoch 0', fontsize=20, pad=30)  # Add a title with styling
    plt.savefig('3dEmbedEpoch0.png', dpi=dpi)

    """Animate evolution of 3D embeddings"""


    def animate3DEmbed(i):
        embed = embeddings[i].detach().cpu().numpy()
        ax.clear()
        ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
                   s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)
        plt.title(f'3D Node Embedding\nEpoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i] * 100:.2f}%',
                  fontsize=18, pad=40)


    # Create figure
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    ax = fig.add_subplot(projection='3d')
    # Set plot configurations
    ax.set_box_aspect([1, 1, 1])  # Maintain an equal aspect ratio for all axes
    # Create animation
    anim = animation.FuncAnimation(fig, animate3DEmbed, np.arange(0, epoch, epochGran), interval=800, repeat=True)
    animName = "train3DEmbedAnim"
    # Save animation to video (optional)
    anim.save(animName + ".mp4", writer="ffmpeg")
    # Or to GIF instead
    anim.save(animName + ".gif", writer="pillow")

    toc = time.time()
    print("All simulations completed. Program terminating. Total time taken was",
          str(datetime.timedelta(seconds=toc - tic)))
