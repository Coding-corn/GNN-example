from collections import Counter
from datetime import datetime

import networkx as nx
from matplotlib import pyplot as plt, animation
import torch
from tqdm import tqdm


def plot_dist(data, dpi=96):
    """
    Plot bar chart of total number of interactions wrt node
    :param data: data object
    :param dpi: dpi of the plot
    :return:
    """
    # Create a histogram of user interactions with items across all time stamps
    plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
    plt.bar(*torch.unique(data.src, return_counts=True), label='User Interactions')
    plt.bar(*torch.unique(data.dst, return_counts=True), label='Item Interactions', color=['forestgreen'])
    plt.xlim(0, data.num_nodes - 1)
    plt.ylabel(ylabel="Count")
    plt.xlabel(xlabel="Index")
    plt.legend(loc='best')
    plt.grid(True, which="both", ls=":")
    plt.title('User-Item Interaction Distribution')
    plt.tight_layout()
    plt.savefig('interDist.png', bbox_inches="tight", dpi=dpi)
    plt.show()


def anim(data, k=5, time_gran='days', dpi=96):
    """
    Create and save animation of interactions between top k users and items
    :param data: data object
    :param k: top k users with the most number of interactions
    :param time_gran: smallest time denomination
    :param dpi: dpi of the plot
    :return:
    """
    # Note that t is in seconds. At a single timestamp, 2 or more interactions can occur simultaneously
    src, dst, t = data.src, data.dst, data.t

    # Find the top k users with the most number of interactions
    u = torch.topk(torch.unique(src, return_counts=True)[1], k).indices.tolist()
    # Create boolean mask: which src indices are in u
    mask = torch.isin(src, torch.tensor(u))
    # Ensure the number of user/ item nodes does not exceed a certain amount so that the plot does not become too clustered
    if len(u) > 50 or len(torch.unique(dst[mask])) > 50:
        raise Exception(
            f'Too many users/ items to plot. Expected less than 50 users/ items. Got {len(u)} users and {len(torch.unique(dst[mask]))} items')
    print(
        f'Users: {len(u)} | Items: {len(torch.unique(dst[mask]))} | Timestamps: {len(torch.unique(t[mask]))} | Interactions: {mask.sum()}')
    u_i_t = torch.concat((src[:, None], dst[:, None], t[:, None]), 1)[mask]
    if time_gran == 'seconds':
        # t is already in seconds
        t_con = u_i_t[:, 2]
    elif time_gran == 'minutes':
        # Convert seconds to hours; serves to index the rows of u_i_t according to minute
        t_con = u_i_t[:, 2] // 60
    elif time_gran == 'hours':
        # Convert seconds to hours; serves to index the rows of u_i_t according to hour
        t_con = u_i_t[:, 2] // (60 * 60)
    elif time_gran == 'days':
        # Convert seconds to days; serves to index the rows of u_i_t according to the day
        t_con = u_i_t[:, 2] // (60 * 60 * 24)
    else:
        raise ValueError(f'Invalid time granularity: {time_gran}')
    # Each key in the dictionary corresponds to the chosen denomination of time, while the values correspond to Tensors containing the user and item nodes
    t_conv_dict = {t.item(): u_i_t[t_con == t, :2] for t in torch.unique(t_con)}

    # Initialise bipartite graph
    B = nx.Graph()
    B.add_nodes_from(u, bipartite=0)
    B.add_nodes_from([i.item() for i in torch.unique(dst[mask])], bipartite=1)
    B.add_edges_from(set([(u.item(), i.item()) for u, i, t in u_i_t]))
    # Get a bipartite layout
    pos = nx.bipartite_layout(B, u)
    del u_i_t, mask

    def animate(idx):
        ax.cla()  # Clear the current axes
        ax.axis('off')

        # Draw nodes, edges, and node labels
        nx.draw_networkx_nodes(B, pos, node_color=['#76b900' if n in u else '#3b5998' for n in B.nodes()], ax=ax)
        nx.draw_networkx_edges(B, pos, edgelist=B.edges(), edge_color='black', ax=ax)
        nx.draw_networkx_labels(B, pos, ax=ax)

        # Label the active edges (red) with the number of interaction(s) within the respective interval of time
        # Count occurrences of each edge
        edge_counter = Counter([(u.item(), i.item()) for u, i in t_conv_dict[idx]])
        # Build a unique edge list for NetworkX (since nx.Graph uses simple edges)
        edgelist = list(edge_counter)
        # Prepare edge_labels: edge -> count
        edge_labels = {edge: edge_counter[edge] for edge in edgelist}
        # Highlight the currently active interaction(s) in red to distinguish it from the inactive interactions in black
        # Note that at a single timestamp, 2 or more interactions could possibly occur
        nx.draw_networkx_edges(B, pos, edgelist=edgelist, edge_color='red',
                               width=2.0, ax=ax)
        nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, font_color='blue', ax=ax)

        if time_gran == 'seconds':
            ax.set_title(f'Seconds: {str(datetime.timedelta(seconds=idx))}', fontsize=18, pad=20)
        elif time_gran == 'minutes':
            ax.set_title(f'Minutes: {idx // (24 * 60):02}:{(idx % (24 * 60)) // 60:02}:{idx % 60}:00', fontsize=18,
                         pad=20)
        elif time_gran == 'hours':
            ax.set_title(f'Hours: {idx // 24:02}:{idx % 24:02}:00:00', fontsize=18, pad=20)
        else:
            ax.set_title(f'Days: {idx}:00:00:00', fontsize=18, pad=20)
        fig.tight_layout()

    def progress_callback(current_frame, total_frames):
        # tqdm uses 0-based, so add 1 to avoid off-by-one errors
        pbar.n = current_frame + 1
        pbar.refresh()

    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    anim = animation.FuncAnimation(fig, animate, frames=list(t_conv_dict.keys()), interval=125, repeat=True)
    # Create tqdm progress bar with the total number of frames
    pbar = tqdm(total=len(t_conv_dict.keys()), desc="Frames", ncols=100)
    # Pass the progress_callback to save; tqdm will close itself after
    anim.save("user-item.gif", writer="pillow", progress_callback=progress_callback)
    pbar.close()
