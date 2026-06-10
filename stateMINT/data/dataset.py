import grain.python as grain


class DataSource(grain.RandomAccessDataSource):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def make_loader(
    data: list[dict],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    drop_remainder: bool = True,
) -> grain.DataLoader:
    data_source = DataSource(data)

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed,
    )

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=num_workers,
    )
    return loader
