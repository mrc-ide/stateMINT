from stateMINT.data import get_input_size, STATIC_COVARS
import pytest


@pytest.mark.parametrize(
    "use_cyclical_time, expected_size", [(True, len(STATIC_COVARS) + 4), (False, len(STATIC_COVARS) + 3)]
)
def test_get_input_size(use_cyclical_time, expected_size: int):
    input_size = get_input_size(use_cyclical_time)
    assert input_size == expected_size
