import numpy as np

from stateMINT.data.dataset import DataSource, make_loader


def _records(n):
    return [{"x": np.zeros((4, 3), dtype=np.float32), "y": np.zeros(4, dtype=np.float32)} for _ in range(n)]


def test_datasource_len_and_getitem():
    data = _records(3)
    src = DataSource(data)
    assert len(src) == 3
    assert src[1] is data[1]


def test_loader_batches_and_drops_remainder():
    loader = make_loader(_records(5), batch_size=2, drop_remainder=True)
    batches = list(loader)
    assert len(batches) == 2  # 5 // 2, remainder dropped
    assert batches[0]["x"].shape == (2, 4, 3)


def test_loader_keeps_remainder_when_disabled():
    loader = make_loader(_records(5), batch_size=2, drop_remainder=False)
    sizes = [b["x"].shape[0] for b in make_loader(_records(5), batch_size=2, drop_remainder=False)]
    assert sizes == [2, 2, 1]
    del loader
