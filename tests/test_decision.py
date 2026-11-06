import pytest

from hcc_he_subtyping.decision import fuse_binary_predictions


@pytest.mark.parametrize(
    ("model1", "model2", "expected"),
    [(0, 0, "M1"), (0, 1, "M2"), (1, 0, "M2"), (1, 1, "M3")],
)
def test_fusion_rule(model1, model2, expected):
    assert fuse_binary_predictions(model1, model2) == expected

