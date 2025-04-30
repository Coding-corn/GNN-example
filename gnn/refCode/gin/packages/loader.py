from torch_geometric.loader import DataLoader


def loader(dataset, train_share=0.8, val_share=0.1, batch_size=64):
    # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset) * train_share)]
    val_dataset = dataset[int(len(dataset) * train_share):int(len(dataset) * (train_share + val_share))]
    test_dataset = dataset[int(len(dataset) * (train_share + val_share)):]

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
