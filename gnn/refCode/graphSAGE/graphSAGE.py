import datetime
import time

from gnn.refCode.graphSAGE.packages.models import *

torch.manual_seed(42)
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 24})
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree
from collections import Counter

if __name__ == '__main__':
    epoch = 200
    # Defines the number of epochs between every animation frame
    epochGran = 10
    dpi = 96

    tic = time.time()

    """Load dataset and create batches with neighbour sampling."""
    dataset = Planetoid(root='.', name="Pubmed")
    data = dataset[0]

    # Create batches with neighbour sampling
    train_loader = NeighborLoader(
        data,
        num_neighbors=[5, 10],
        # Total number of subgraphs generated is equivalent to the number of input_nodes divided by batch_size
        batch_size=16,
        input_nodes=data.train_mask,
    )

    """Analyse subgraphs and plot them."""
    # Save every subgraph to a dictionary for later analysis
    subgraphDict = {}
    # Plot each subgraph
    fig = plt.figure(figsize=(16, 16))
    for idx, (subgraph, pos) in enumerate(zip(train_loader, [221, 222, 223, 224])):
        subgraphDict[idx] = subgraph
        G = to_networkx(subgraph, to_undirected=True)
        ax = fig.add_subplot(pos)
        ax.set_title(f'Subgraph {idx}')
        plt.axis('off')
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=True,
                         node_size=200,
                         node_color=subgraph.y,
                         cmap="cool",
                         font_size=10
                         )
    plt.savefig('subgraphs.png')
    plt.show()


    def plot_degree(data):
        # Get ndarry of degrees for each node
        degrees = degree(data.edge_index[0]).numpy()
        # Count the number of nodes for each degree
        numbers = Counter(degrees)

        # Bar plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Number of Nodes')
        plt.bar(numbers.keys(),
                numbers.values(),
                color='#0A047A')


    # Plot node degrees from the original graph
    plot_degree(data)
    plt.title("Main Graph")
    plt.savefig('nodeDegreeMainGraph.png')
    plt.show()

    # Plot node degrees for all subgraphs
    for i in range(len(subgraphDict)):
        subgraph = subgraphDict[i]
        plot_degree(subgraph)
        plt.title("Subgraph " + str(i))
        plt.savefig('nodeDegreeSubgraph' + str(i) + '.png')
        plt.show()

    """Train and test models"""
    timeDict = {'GraphSAGE': 0.0, 'GCN': 0.0, 'GAT': 0.0}
    tic_ = time.time()
    # Create GraphSAGE
    graphsage = GraphSAGE(dataset.num_features, 64, dataset.num_classes)
    # Train model either by using passing loader or dictionary as argument
    graphsage.fit(train_loader, epoch, epochGran)
    toc_ = time.time()
    timeDict['GraphSAGE'] = toc_ - tic_
    # Test
    print(
        f'\nGraphSAGE test accuracy: {test(graphsage, data) * 100:.2f}% | Time Taken: {datetime.timedelta(seconds=timeDict['GraphSAGE'])}\n')

    tic_ = time.time()
    # Create GCN
    gcn = GCN(dataset.num_features, 64, dataset.num_classes)
    # Train
    gcn.fit(data, epoch, epochGran)
    toc_ = time.time()
    timeDict['GCN'] = toc_ - tic_
    # Test
    print(
        f'\nGCN test accuracy: {test(gcn, data) * 100:.2f}% | Time Taken: {datetime.timedelta(seconds=timeDict['GCN'])}\n')

    tic_ = time.time()
    # Create GAT
    gat = GAT(dataset.num_features, 64, dataset.num_classes)
    # Train
    gat.fit(data, epoch, epochGran)
    toc_ = time.time()
    timeDict['GAT'] = toc_ - tic_
    # Test
    print(
        f'\nGAT test accuracy: {test(gat, data) * 100:.2f}% | Time Taken: {datetime.timedelta(seconds=timeDict['GAT'])}\n')

    # Create bar chart of training time among the three GNN models
    plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
    plt.bar(timeDict.keys(), timeDict.values())
    plt.ylabel(ylabel="Time [s]")
    plt.yscale('log')
    plt.title('Training Time Distribution')
    plt.grid(True, which="both", ls=":")
    plt.savefig('trainTime.png', bbox_inches="tight", dpi=dpi)
    plt.show()

    toc = time.time()
    print("All simulations completed. Program terminating. Total time taken was",
          str(datetime.timedelta(seconds=toc - tic)))
