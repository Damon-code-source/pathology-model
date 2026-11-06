import torch

from hcc_he_subtyping.model import CLAMSB


def test_model_output_shapes():
    model = CLAMSB(dropout=True).eval()
    logits, probabilities, prediction, attention, _ = model(torch.randn(13, 768))
    assert logits.shape == (1, 2)
    assert probabilities.shape == (1, 2)
    assert prediction.shape == (1,)
    assert attention.shape == (1, 13)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(1))

