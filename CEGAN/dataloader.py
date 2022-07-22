from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

# -------------Dataloader similar to CGCNN code---------------------


def get_train_val_test_loader(
    dataset,
    collate_fn=default_collate,
    batch_size=64,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    num_workers=1,
    pin_memory=False,
    **kwargs,
):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = dataset.size

    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(
                f'[Warning] train_ratio is None, using 1 - val_ratio - '
                f'test_ratio = {train_ratio} as training data.'
            )
        else:
            print(train_ratio + val_ratio + test_ratio)
            assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))
    if kwargs['train_size'] is not None:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size'] is not None:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size'] is not None:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])

    if test_size == 0:
        val_sampler = SubsetRandomSampler(indices[-valid_size:])

    else:
        val_sampler = SubsetRandomSampler(
            indices[-(valid_size + test_size) : -test_size]
        )
        test_sampler = SubsetRandomSampler(indices[-test_size:])

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if test_size == 0:
        test_loader = []
    else:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
