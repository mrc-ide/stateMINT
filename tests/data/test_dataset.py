from stateMINT.data.dataset import DataSource, make_loader


def test_datasource_len_and_getitem(loader_records_factory):
    data = loader_records_factory(3)
    src = DataSource(data)
    assert len(src) == 3
    assert src[1] is data[1]


def test_loader_batches_and_drops_remainder(loader_records_factory):
    loader = make_loader(loader_records_factory(5), batch_size=2, drop_remainder=True)
    batches = list(loader)
    assert len(batches) == 2  # 5 // 2, remainder dropped
    assert batches[0]["x"].shape == (2, 4, 3)


def test_loader_keeps_remainder_when_disabled(loader_records_factory):
    loader = make_loader(loader_records_factory(5), batch_size=2, drop_remainder=False)
    sizes = [b["x"].shape[0] for b in loader]
    assert sizes == [2, 2, 1]
