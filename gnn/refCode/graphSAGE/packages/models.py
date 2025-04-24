from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv
import torch.nn.functional as F
import torch


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, batchLoader, epoch: int, epochGran: int):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        for epoch_ in range(epoch + 1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            if isinstance(batchLoader, dict):
                batch = batchLoader.values()
            elif isinstance(batchLoader, NeighborLoader):
                batch = batchLoader
            else:
                raise TypeError(f"batchLoader must be dict or NeighborLoader, got {type(batchLoader)}")
            for batch_ in batch:
                optimizer.zero_grad()
                out = self(batch_.x, batch_.edge_index)
                loss = criterion(out[batch_.train_mask], batch_.y[batch_.train_mask])
                total_loss += loss
                acc += accuracy(out[batch_.train_mask].argmax(dim=1), batch_.y[batch_.train_mask])
                loss.backward()
                optimizer.step()

                # Validation
                val_loss += criterion(out[batch_.val_mask], batch_.y[batch_.val_mask])
                val_acc += accuracy(out[batch_.val_mask].argmax(dim=1), batch_.y[batch_.val_mask])

            # Print metrics every epochGran epochs
            if (epoch_ % epochGran == 0):
                print(f'Epoch {epoch_:>3} | Train Loss: {total_loss / len(batchLoader):.3f} '
                      f'| Train Acc: {acc / len(batchLoader) * 100:>6.2f}% | Val Loss: '
                      f'{val_loss / len(batchLoader):.2f} | Val Acc: '
                      f'{val_acc / len(batchLoader) * 100:.2f}%')


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=heads)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epoch: int, epochGran: int):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        for epoch_ in range(epoch + 1):
            # Training
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

            # Print metrics every epochGran epochs
            if (epoch_ % epochGran == 0):
                print(f'Epoch {epoch_:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc * 100:.2f}%')


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
        return F.log_softmax(h, dim=1)

    def fit(self, data, epoch: int, epochGran: int):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        for epoch_ in range(epoch + 1):
            # Training
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1),
                           data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                               data.y[data.val_mask])

            # Print metrics every epochGran epochs
            if (epoch_ % epochGran == 0):
                print(f'Epoch {epoch_:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc * 100:.2f}%')


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def test(model, data):
    """Evaluate the model on the test set and print the accuracy score."""
    model.eval()
    out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc
