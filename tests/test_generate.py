from pathlib import Path

import pytest
import torch

from examples.causal_softmax import spec as causal_softmax_spec
from examples.relu import spec as relu_spec
from miniattn.generate import generate
from miniattn.ref import ref
from miniattn.spec import AttentionSpec, Op, causal, identity, relu, scale, softmax


ROOT = Path(__file__).resolve().parents[1]


def load_attention(source):
    namespace = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    return namespace["attention"]


def test_generated_causal_softmax_matches_reference():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 17, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = load_attention(generate(causal_softmax_spec, 4, 3))(q, k, v)
    torch.testing.assert_close(actual, ref(q, k, v), rtol=1e-5, atol=1e-6)


def test_generated_relu_supports_rectangular_inputs():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 7, 5)
    k = torch.randn(1, 2, 11, 5)
    v = torch.randn(1, 2, 11, 3)

    scores = q @ k.transpose(-2, -1)
    scores = torch.relu(scores * q.shape[-1] ** -0.5)
    expected = scores @ v
    actual = load_attention(generate(relu_spec, 4, 3))(q, k, v)

    assert actual.shape == (1, 2, 7, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_generated_composes_primitives_without_variant_name():
    spec = AttentionSpec(
        name="unseen_combination",
        pattern="parallel",
        score_mod=(scale(0.125), relu()),
        mask_mod=causal(),
        rownorm=identity(),
    )
    torch.manual_seed(2)
    q = torch.randn(1, 1, 17, 5)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 17, 3)

    scores = torch.relu((q @ k.transpose(-2, -1)) * 0.125)
    indices = torch.arange(17)
    scores = scores.masked_fill(indices[None, :] > indices[:, None], 0.0)
    expected = scores @ v
    actual = load_attention(generate(spec, 4, 3))(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tile_q, tile_k", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_generate_rejects_invalid_tiles(tile_q, tile_k):
    with pytest.raises(ValueError):
        generate(causal_softmax_spec, tile_q, tile_k)


def test_generate_rejects_unknown_primitive():
    spec = AttentionSpec(
        name="invalid",
        pattern="parallel",
        score_mod=(Op("sigmoid"),),
        mask_mod=None,
        rownorm=identity(),
    )
    with pytest.raises(ValueError):
        generate(spec, 4, 3)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 1, 7, 4), (1, 1, 7, 5), (1, 1, 7, 3)),
        ((1, 1, 7, 4), (1, 1, 8, 4), (1, 1, 7, 3)),
        ((1, 1, 7, 4), (1, 1, 8, 4), (1, 1, 8, 3)),
    ],
)
def test_generated_causal_rejects_invalid_shapes(shapes):
    q, k, v = (torch.randn(shape) for shape in shapes)
    attention = load_attention(generate(causal_softmax_spec, 4, 3))

    with pytest.raises(ValueError):
        attention(q, k, v)


@pytest.mark.parametrize(
    ("module_name", "output_name"),
    [
        ("examples.causal_softmax", "causal_softmax.py"),
        ("examples.relu", "relu.py"),
    ],
)
def test_committed_generated_examples_are_current(module_name, output_name):
    module = __import__(module_name, fromlist=["spec"])
    generated = (ROOT / "generated" / output_name).read_text(encoding="utf-8")
    assert generated == generate(module.spec, 64, 64)
    assert "miniattn" not in generated
