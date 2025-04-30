import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx
from matplotlib import animation


def plot_cat_graph(dataset, model, suptitle, savefig, nrows=4, ncols=4, size=4):
    """
    Function to display graph classification results of the model
    :param dataset: dataset object
    :param model: model object
    :param suptitle: title of the plot
    :param savefig: name of the file to save the plot to
    :param nrows: number of rows in the plot
    :param ncols: number of columns in the plot
    :param size: width and height of each subplot in inches
    :return:
    """
    # Set font size to 24 and figure DPI to 300 for everything in this script
    with plt.rc_context({'font.size': 24, 'figure.dpi': 300}):
        fig, ax = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
        fig.suptitle(suptitle)

        for i, data in enumerate(dataset[len(dataset) - nrows * ncols:]):
            # Calculate colour (green if correct, red otherwise)
            _, out = model(data.x, data.edge_index, data.batch)
            color = "green" if out.argmax(dim=1) == data.y else "red"

            # Plot graph
            ix = np.unravel_index(i, ax.shape)
            ax[ix].axis('off')
            G = to_networkx(data, to_undirected=True)
            nx.draw_networkx(G,
                             pos=nx.spring_layout(G, seed=0),
                             with_labels=False,
                             node_size=150,
                             node_color=color,
                             width=0.8,
                             ax=ax[ix]
                             )
        fig.tight_layout()
        plt.savefig(savefig + '.png')
        plt.show()


def anim_graph(dataset, start=0, end=10):
    def animate(i):
        # Setup
        ax.cla()  # Clear the current axes
        grey = (0.92, 0.92, 0.92, 1.0)
        ax.xaxis.set_pane_color(grey)
        ax.yaxis.set_pane_color(grey)
        ax.zaxis.set_pane_color(grey)

        G = to_networkx(dataset[i], to_undirected=True)
        # 3D spring layout
        pos = nx.spring_layout(G, dim=3, seed=0)
        # Extract node and edge positions from the layout
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        # Plot the nodes - "depth" scales alpha automatically
        ax.scatter(*node_xyz.T, s=500, c="#0A047A")
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")
        ax.set_title(f'Graph {i}', fontsize=24, pad=20)
        fig.tight_layout()

    # Create the 3D figure
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(start, end), interval=500)
    animName = "graphs"
    anim.save(animName + ".mp4", writer="ffmpeg")
    anim.save(animName + ".gif", writer="pillow")
