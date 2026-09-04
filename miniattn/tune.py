import math
import statistics
import time
from collections.abc import Callable, Iterable

import torch

from .generate import generate
from .spec import AttentionSpec


DEFAULT_TILE_CANDIDATES = (16, 32, 64, 128, 256, 512, 1024)


def measure_latency(fn: Callable[[], object], *, warmups: int, repeats: int) -> float:
    if not callable(fn):
        raise ValueError("fn must be callable")
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    with torch.inference_mode():
        for _ in range(warmups):
            fn()
        samples = []
        for _ in range(repeats):
            begin = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - begin) * 1000.0)
    return statistics.median(samples)


def _validate_tensors(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    tensors = (q, k, v)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("q, k, and v must be tensors")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("q, k, and v must be 4D tensors")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("q, k, and v must be CPU tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("q, k, and v must use torch.float32")
    if any(0 in tensor.shape for tensor in tensors):
        raise ValueError("q, k, and v must have non-empty dimensions")
    if not (q.shape[:2] == k.shape[:2] == v.shape[:2]):
        raise ValueError("q, k, and v must have matching batch and head dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length")


def _normalize_candidates(
    candidates: Iterable[int], dimension: int, name: str
) -> tuple[int, ...]:
    try:
        values = list(candidates)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of positive integers") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must contain only positive integers")
    return tuple(sorted({value for value in values if value <= dimension} | {dimension}))


def _kv_tile_interactions(
    spec: AttentionSpec, sq: int, sk: int, tile_q: int, tile_k: int
) -> int:
    if spec.mask_mod is not None and spec.mask_mod.name == "causal":
        total = 0
        for q_start in range(0, sq, tile_q):
            q_end = min(q_start + tile_q, sq)
            total += math.ceil(q_end / tile_k)
        return total
    return math.ceil(sq / tile_q) * math.ceil(sk / tile_k)


def tune(
    spec: AttentionSpec,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    tile_q_candidates: Iterable[int] = DEFAULT_TILE_CANDIDATES,
    tile_k_candidates: Iterable[int] = DEFAULT_TILE_CANDIDATES,
    warmups: int = 5,
    repeats: int = 20,
) -> tuple[list[dict[str, float | int | bool]], dict[str, float | int | bool]]:
    _validate_tensors(q, k, v)
    if not isinstance(spec, AttentionSpec):
        raise ValueError("spec must be an AttentionSpec")
    if not callable(reference_fn):
        raise ValueError("reference_fn must be callable")
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    sq = q.shape[-2]
    sk = k.shape[-2]
    tile_qs = _normalize_candidates(tile_q_candidates, sq, "tile_q_candidates")
    tile_ks = _normalize_candidates(tile_k_candidates, sk, "tile_k_candidates")

    with torch.inference_mode():
        expected = reference_fn(q, k, v)

    batch, heads = q.shape[:2]
    results = []
    for tile_q in tile_qs:
        for tile_k in tile_ks:
            source = generate(spec, tile_q, tile_k)
            namespace = {}
            exec(compile(source, "<generated>", "exec"), namespace)
            attention = namespace["attention"]

            with torch.inference_mode():
                actual = attention(q, k, v)
            if actual.shape != expected.shape:
                raise AssertionError(
                    f"correctness failed for spec={spec.name}, tile_q={tile_q}, "
                    f"tile_k={tile_k}, shape={tuple(q.shape)}/{tuple(k.shape)}/"
                    f"{tuple(v.shape)}: output shape {tuple(actual.shape)} != "
                    f"{tuple(expected.shape)}"
                )
            max_error = (actual - expected).abs().max().item()
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                raise AssertionError(
                    f"correctness failed for spec={spec.name}, tile_q={tile_q}, "
                    f"tile_k={tile_k}, shape={tuple(q.shape)}/{tuple(k.shape)}/"
                    f"{tuple(v.shape)}, max_abs_error={max_error:.3e}"
                )

            latency_ms = measure_latency(
                lambda attention=attention: attention(q, k, v),
                warmups=warmups,
                repeats=repeats,
            )
            calls_per_second = 1000.0 / latency_ms
            score_tile_elements = (
                batch
                * heads
                * min(tile_q, sq)
                * min(tile_k, sk)
            )
            score_tile_bytes = score_tile_elements * q.element_size()
            results.append(
                {
                    "tile_q": tile_q,
                    "tile_k": tile_k,
                    "median_latency_ms": latency_ms,
                    "calls_per_second": calls_per_second,
                    "tokens_per_second": batch * sq * calls_per_second,
                    "score_tile_elements": score_tile_elements,
                    "score_tile_bytes": score_tile_bytes,
                    "score_tile_kib": score_tile_bytes / 1024.0,
                    "q_tiles": math.ceil(sq / tile_q),
                    "kv_tile_interactions": _kv_tile_interactions(
                        spec, sq, sk, tile_q, tile_k
                    ),
                    "is_best": False,
                }
            )

    best = min(results, key=lambda result: result["median_latency_ms"])
    best["is_best"] = True
    return results, best
