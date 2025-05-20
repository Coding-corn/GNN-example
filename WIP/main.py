# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighbourhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper, we
# decided to present this version here as it is more realistic and a better
# test bed for future methods.

import os.path as osp
import time

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from packages.utils import *

if __name__ == '__main__':
    # TODO Create seed for random, np and torch for reproducibility of results
    dpi = 96

    # TODO Revert
    epoch = 50
    # epoch = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO Maybe swap to a smaller dataset, after fully understanding the paper and data set used
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
    dataset = JODIEDataset(path, name='Wikipedia')
    data = dataset[0]

    # Note that t is in seconds. At a single timestamp, 2 or more interactions can occur simultaneously
    src, dst, t, msg = data.src, data.dst, data.t, data.msg
    # TODO The following analysis may be too task specific to the Wikipedia dataset. Modifications may be needed for other datasets
    # TODO Revert
    # plot_dist(data)

    # TODO Revert
    # anim(data)

    # For small datasets, we can put the whole dataset on GPU and thus avoid
    # expensive memory transfer costs for mini-batches:
    data = data.to(device)

    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

    train_loader = TemporalDataLoader(
        train_data,
        batch_size=200,
        neg_sampling_ratio=1.0,
    )
    val_loader = TemporalDataLoader(
        val_data,
        batch_size=200,
        neg_sampling_ratio=1.0,
    )
    test_loader = TemporalDataLoader(
        test_data,
        batch_size=200,
        neg_sampling_ratio=1.0,
    )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)


    class GraphAttentionEmbedding(torch.nn.Module):
        def __init__(self, in_channels, out_channels, msg_dim, time_enc):
            super().__init__()
            self.time_enc = time_enc
            edge_dim = msg_dim + time_enc.out_channels
            self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                        dropout=0.1, edge_dim=edge_dim)

        def forward(self, x, last_update, edge_index, t, msg):
            rel_t = last_update[edge_index[0]] - t
            rel_t_enc = self.time_enc(rel_t.to(x.dtype))
            edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
            return self.conv(x, edge_index, edge_attr)


    class LinkPredictor(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.lin_src = Linear(in_channels, in_channels)
            self.lin_dst = Linear(in_channels, in_channels)
            self.lin_final = Linear(in_channels, 1)

        def forward(self, z_src, z_dst):
            h = self.lin_src(z_src) + self.lin_dst(z_dst)
            h = h.relu()
            return self.lin_final(h)


    memory_dim = time_dim = embedding_dim = 100

    # TODO Print summary of memory, gnn and link_pred modules using torch_geometric.nn.summary
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


    def train():
        memory.train()
        gnn.train()
        link_pred.train()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)

            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                    data.msg[e_id].to(device))
            pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            # Update memory and neighbour loader with ground-truth state.
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

            loss.backward()
            optimizer.step()
            memory.detach()
            total_loss += float(loss) * batch.num_events

        return total_loss / train_data.num_events


    @torch.no_grad()
    def test(loader):
        memory.eval()
        gnn.eval()
        link_pred.eval()

        torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(device)

            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                    data.msg[e_id].to(device))
            pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)),
                 torch.zeros(neg_out.size(0))], dim=0)

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
        return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


    tic = time.time()

    loss, val_ap, val_auc, test_ap, test_auc = [], [], [], [], []
    # TODO Replace with tqdm, and remove printed output
    for epoch_ in range(epoch):
        # TODO Move testing regime outside of training epoch loop. Specify if best checkpoint should be used
        # TODO Save checkpoint of model parameters after every training epoch
        loss_ = train()
        # TODO Unify displayed values to a single line for every epoch, instead of 3 separate lines
        print(f'Epoch: {epoch_:02d}, Loss: {loss_:.4f}')
        val_ap_, val_auc_ = test(val_loader)
        test_ap_, test_auc_ = test(test_loader)
        print(f'Val AP: {val_ap_:.4f}, Val AUC: {val_auc_:.4f}')
        print(f'Test AP: {test_ap_:.4f}, Test AUC: {test_auc_:.4f}')

        loss.append(loss_)
        val_ap.append(val_ap_)
        val_auc.append(val_auc_)
        test_ap.append(test_ap_)
        test_auc.append(test_auc_)

    df = pd.DataFrame({"epoch": range(epoch), "loss": loss, "val_ap": val_ap, "val_auc": val_auc, "test_ap": test_ap,
                       "test_auc": test_auc})
    df.to_csv('result.csv', index=False)

    # TODO May need to have separate plots for training and validation/testing
    plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
    plt.plot(df["epoch"], df["loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_ap"], label="Val AP")
    plt.plot(df["epoch"], df["val_auc"], label="Val AUC")
    plt.plot(df["epoch"], df["test_ap"], label="Test AP")
    plt.plot(df["epoch"], df["test_auc"], label="Test AUC")
    plt.legend(loc='best')
    plt.grid(True, which="both", ls=":")
    plt.xlabel(xlabel="Epoch")
    plt.xlim(df["epoch"].iloc[0], df["epoch"].iloc[-1])
    # TODO May have to modify ylim once graphs have been separated accordingly
    plt.ylim(
        min(min(df["loss"]), min(df["val_ap"]), min(df["val_auc"]), min(df["test_ap"]), min(df["test_auc"])),
        max(max(df["loss"]), max(df["val_ap"]), max(df["val_auc"]), max(df["test_ap"]), max(df["test_auc"]))
    )
    plt.title('Performance')
    plt.savefig('result.png', bbox_inches="tight", dpi=dpi)
    plt.show()

    toc = time.time()
    print("All simulations completed. Program terminating. Total time taken was",
          str(datetime.timedelta(seconds=toc - tic)))
