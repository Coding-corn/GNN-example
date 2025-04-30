import pandas as pd
from matplotlib import pyplot as plt

from .models import *


def train(model, train_loader, val_loader, test_loader, epoch, epoch_gran):
    df = pd.DataFrame({"epoch": range(epoch + 1)})
    total_loss, acc = [], []
    val_loss, val_acc = [], []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=0.01)

    model.train()
    for epoch_ in df['epoch']:
        total_loss_ = 0
        acc_ = 0

        # Train on batches
        for data in train_loader:
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss_ += loss / len(train_loader)
            acc_ += accuracy(out.argmax(dim=1), data.y) / len(train_loader)
            loss.backward()
            optimizer.step()
        total_loss.append(total_loss_.item())
        acc.append(acc_)

        # Validation
        val_loss_, val_acc_ = test(model, val_loader)
        val_loss.append(val_loss_.item())
        val_acc.append(val_acc_)

        # Print metrics every epoch_gran number of epochs
        if (epoch_ % epoch_gran == 0):
            print(f'Epoch {epoch_:>3} | Train Loss: {total_loss_:.2f} '
                  f'| Train Acc: {acc_ * 100:>5.2f}% '
                  f'| Val Loss: {val_loss_:.2f} '
                  f'| Val Acc: {val_acc_ * 100:.2f}%')

    # Save training and validation curves in csv and png
    df['train_loss'], df['train_acc'] = total_loss, acc
    df['val_loss'], df['val_acc'] = val_loss, val_acc
    dpi = 96
    plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
    plt.plot(df['epoch'], total_loss, label="Train loss")
    plt.plot(df['epoch'], acc, label="Train acc")
    plt.plot(df['epoch'], val_loss, label="Val loss")
    plt.plot(df['epoch'], val_acc, label="Val acc")
    plt.legend(loc='best')
    plt.grid(True, which="both", ls=":")
    plt.xlabel(xlabel="Epoch")
    plt.xlim(df['epoch'].iloc[0], df['epoch'].iloc[-1])
    if isinstance(model, GCN):
        df.to_csv('gcnTrainVal.csv', index=False)
        plt.title('GCN Training and Validation Curves')
        plt.savefig('gcnTrainVal.png', bbox_inches="tight", dpi=dpi)
    elif isinstance(model, GIN):
        df.to_csv('ginTrainVal.csv', index=False)
        plt.title('GIN Training and Validation Curves')
        plt.savefig('ginTrainVal.png', bbox_inches="tight", dpi=dpi)
    else:
        raise TypeError(f"model must be GCN or GIN, got {type(model)}")
    plt.show()

    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%\n')

    return model


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()
