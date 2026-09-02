import pytest
import torch

from miniattn.online import online
from miniattn.ref import ref


@pytest.mark.parametrize("tile_q", [1, 4, 8, 32])
@pytest.mark.parametrize("tile_k", [1, 3, 8, 32])
def test_online_matches_reference(tile_q, tile_k):
    torch.manual_seed(0)
    q = torch.randn(2, 3, 17, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = ref(q, k, v)
    actual = online(q, k, v, tile_q, tile_k)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_online_is_stable_for_large_scores():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 17, 8) * 100
    k = torch.randn_like(q) * 100
    v = torch.randn_like(q)

    actual = online(q, k, v, tile_q=4, tile_k=3)
    expected = ref(q, k, v)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("tile_q", "tile_k"),
    [
        (0, 1),
        (-1, 1),
        (1.5, 1),
        (1, 0),
        (1, -1),
        (1, 1.5),
    ],
)
def test_invalid_tiles(tile_q, tile_k):
    q = torch.randn(1, 1, 4, 2)

    with pytest.raises(ValueError):
        online(q, q, q, tile_q, tile_k)
