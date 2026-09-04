import importlib

import pytest
import torch

from examples.causal_softmax import spec as causal_spec
from examples.relu import spec as relu_spec
from miniattn.ref import ref


tune_module = importlib.import_module("miniattn.tune")


def test_measure_latency_uses_warmups_repeats_and_median(monkeypatch):
    clock = iter((0.0, 0.003, 1.0, 1.001, 2.0, 2.002))
    calls = []
    monkeypatch.setattr(tune_module.time, "perf_counter", lambda: next(clock))

    latency = tune_module.measure_latency(
        lambda: calls.append(True), warmups=2, repeats=3
    )

    assert len(calls) == 5
    assert latency == pytest.approx(2.0)


def test_tune_searches_generated_candidates_and_metrics(monkeypatch):
    torch.manual_seed(0)
    q = torch.randn(1, 1, 7, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    generated = []
    original_generate = tune_module.generate

    def spy_generate(spec, tile_q, tile_k):
        generated.append((tile_q, tile_k))
        return original_generate(spec, tile_q, tile_k)

    monkeypatch.setattr(tune_module, "generate", spy_generate)
    results, best = tune_module.tune(
        causal_spec,
        q=q,
        k=k,
        v=v,
        reference_fn=ref,
        tile_q_candidates=(3,),
        tile_k_candidates=(2,),
        warmups=0,
        repeats=1,
    )

    expected_candidates = [(3, 2), (3, 7), (7, 2), (7, 7)]
    assert generated == expected_candidates
    assert [(r["tile_q"], r["tile_k"]) for r in results] == expected_candidates
    assert best is next(r for r in results if r["is_best"])
    assert best["median_latency_ms"] == min(
        r["median_latency_ms"] for r in results
    )

    partial = next(r for r in results if (r["tile_q"], r["tile_k"]) == (3, 2))
    assert partial["score_tile_elements"] == 6
    assert partial["score_tile_bytes"] == 24
    assert partial["score_tile_kib"] == pytest.approx(24 / 1024)
    assert partial["q_tiles"] == 3
    assert partial["kv_tile_interactions"] == 9


def test_tune_supports_rectangular_relu_inputs():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 5, 3)
    k = torch.randn(1, 1, 7, 3)
    v = torch.randn(1, 1, 7, 2)

    def reference_fn(q, k, v):
        scores = q @ k.transpose(-2, -1)
        scores = torch.relu(scores * q.shape[-1] ** -0.5)
        return scores @ v

    results, best = tune_module.tune(
        relu_spec,
        q=q,
        k=k,
        v=v,
        reference_fn=reference_fn,
        tile_q_candidates=(3,),
        tile_k_candidates=(2,),
        warmups=0,
        repeats=1,
    )

    assert len(results) == 4
    assert best in results
    interactions = {
        (result["tile_q"], result["tile_k"]): result["kv_tile_interactions"]
        for result in results
    }
    assert interactions == {(3, 2): 8, (3, 7): 2, (5, 2): 4, (5, 7): 1}


def test_tune_correctness_failure_is_fail_fast(monkeypatch):
    torch.manual_seed(2)
    q = torch.randn(1, 1, 7, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    generated = []
    original_generate = tune_module.generate

    def spy_generate(spec, tile_q, tile_k):
        generated.append((tile_q, tile_k))
        return original_generate(spec, tile_q, tile_k)

    monkeypatch.setattr(tune_module, "generate", spy_generate)
    with pytest.raises(AssertionError, match="spec=causal_softmax.*max_abs_error"):
        tune_module.tune(
            causal_spec,
            q=q,
            k=k,
            v=v,
            reference_fn=lambda q, k, v: ref(q, k, v) + 1.0,
            tile_q_candidates=(3, 4),
            tile_k_candidates=(2, 4),
            warmups=0,
            repeats=1,
        )
    assert generated == [(3, 2)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_q_candidates": ()},
        {"tile_q_candidates": (0,)},
        {"tile_k_candidates": (-1,)},
        {"warmups": -1},
        {"repeats": 0},
    ],
)
def test_tune_rejects_invalid_arguments(kwargs):
    q = torch.randn(1, 1, 5, 3)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    call_kwargs = {"warmups": 0, "repeats": 1}
    call_kwargs.update(kwargs)
    with pytest.raises(ValueError):
        tune_module.tune(
            causal_spec,
            q=q,
            k=k,
            v=v,
            reference_fn=ref,
            **call_kwargs,
        )


def test_tune_rejects_non_float32_inputs():
    q = torch.randn(1, 1, 5, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(ValueError, match="float32"):
        tune_module.tune(
            causal_spec,
            q=q,
            k=k,
            v=v,
            reference_fn=ref,
            warmups=0,
            repeats=1,
        )
