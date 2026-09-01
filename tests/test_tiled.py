import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from miniattn.ref import ref
from miniattn.tiled import tiled


def test_reference_causal_prefix_mean():
    q = torch.zeros(1, 1, 4, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]]]])
    expected = torch.tensor(
        [[[
            [1.0, 10.0],
            [1.5, 15.0],
            [7.0 / 3.0, 70.0 / 3.0],
            [15.0 / 4.0, 150.0 / 4.0],
        ]]]
    )

    torch.testing.assert_close(ref(q, k, v), expected, rtol=1e-5, atol=1e-6)


def test_reference_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(ref(q, k, v), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "shape, tiles",
    [
        ((1, 1, 7, 4), [1, 3, 7, 8]),
        ((1, 2, 17, 8), [4, 8, 16, 32]),
        ((2, 3, 33, 16), [16, 32]),
    ],
)
def test_tiled_matches_reference(shape, tiles):
    torch.manual_seed(0)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    expected = ref(q, k, v)

    for tile_q in tiles:
        actual = tiled(q, k, v, tile_q)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rejects_non_4d_inputs():
    q = torch.randn(1, 7, 4)
    k = torch.randn(1, 1, 7, 4)
    v = torch.randn_like(k)

    with pytest.raises(ValueError):
        ref(q, k, v)


def test_rejects_unequal_shapes():
    q = torch.randn(1, 1, 7, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn_like(q)

    with pytest.raises(ValueError):
        tiled(q, k, v, 4)


def test_rejects_non_float32_inputs():
    q = torch.randn(1, 1, 7, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        ref(q, q, q)


@pytest.mark.parametrize("tile_q", [0, -1, 1.5])
def test_rejects_invalid_tiles(tile_q):
    q = torch.randn(1, 1, 7, 4)
    with pytest.raises(ValueError):
        tiled(q, q, q, tile_q)
