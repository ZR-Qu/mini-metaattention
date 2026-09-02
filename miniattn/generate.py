import argparse
import importlib
import math
from numbers import Real
from pathlib import Path

from .spec import AttentionSpec, Op


def _validate_spec(spec: AttentionSpec, tile_q: int, tile_k: int) -> None:
    if not isinstance(spec, AttentionSpec):
        raise ValueError("spec must be an AttentionSpec")
    if not isinstance(tile_q, int) or isinstance(tile_q, bool) or tile_q <= 0:
        raise ValueError("tile_q must be a positive integer")
    if not isinstance(tile_k, int) or isinstance(tile_k, bool) or tile_k <= 0:
        raise ValueError("tile_k must be a positive integer")
    if spec.pattern != "parallel":
        raise ValueError("only the parallel pattern is supported")
    if spec.mask_mod is not None and spec.mask_mod.name != "causal":
        raise ValueError("only the causal mask is supported")
    if spec.rownorm.name not in {"softmax", "identity"}:
        raise ValueError("unsupported rownorm primitive")

    for op in spec.score_mod:
        if op.name not in {"scale", "relu"}:
            raise ValueError("unsupported score_mod primitive")
        if op.name == "relu" and op.value is not None:
            raise ValueError("relu does not accept a value")
        if op.name == "scale" and op.value is not None:
            if not isinstance(op.value, Real) or isinstance(op.value, bool):
                raise ValueError("scale value must be numeric")
            if not math.isfinite(float(op.value)):
                raise ValueError("scale value must be finite")


def _score_mod(ops: tuple[Op, ...]) -> str:
    lines = []
    for op in ops:
        if op.name == "scale":
            factor = "q.shape[-1] ** -0.5" if op.value is None else repr(float(op.value))
            lines.append(f"scores = scores * {factor}")
        elif op.name == "relu":
            lines.append("scores = torch.relu(scores)")
    return "\n".join(lines)


def _mask(mask: Op | None, rownorm: Op) -> tuple[str, str, str]:
    if mask is None:
        return "sk", "", ""

    fill = "float(\"-inf\")" if rownorm.name == "softmax" else "0.0"
    setup = "        q_idx = torch.arange(q_start, q_end, device=q.device)"
    body = "\n".join(
        [
            "            k_idx = torch.arange(k_start, k_end, device=q.device)",
            "            causal = k_idx[None, :] > q_idx[:, None]",
            f"            scores = scores.masked_fill(causal, {fill})",
        ]
    )
    return "q_end", setup, body


def _rownorm(rownorm: Op) -> tuple[str, str, str]:
    if rownorm.name == "softmax":
        prologue = "\n".join(
            [
                "        m = torch.full(",
                "            (*q_tile.shape[:-1], 1),",
                '            float("-inf"),',
                "            device=q.device,",
                "            dtype=q.dtype,",
                "        )",
                "        r = torch.zeros_like(m)",
                "        o = q.new_zeros((*q_tile.shape[:-1], v.shape[-1]))",
            ]
        )
        forward = "\n".join(
            [
                "            block_max = scores.max(dim=-1, keepdim=True).values",
                "            m_new = torch.maximum(m, block_max)",
                "            o_scale = torch.exp(m - m_new)",
                "            r = r * o_scale",
                "            scores = torch.exp(scores - m_new)",
                "            r = r + scores.sum(dim=-1, keepdim=True)",
                "            o = o * o_scale + scores @ v_tile",
                "            m = m_new",
            ]
        )
        return prologue, forward, "        out[..., q_start:q_end, :] = o / r"

    prologue = "        o = q.new_zeros((*q_tile.shape[:-1], v.shape[-1]))"
    forward = "            o = o + scores @ v_tile"
    return prologue, forward, "        out[..., q_start:q_end, :] = o"


def generate(spec: AttentionSpec, tile_q: int, tile_k: int) -> str:
    _validate_spec(spec, tile_q, tile_k)
    score_code = _score_mod(spec.score_mod)
    k_stop, mask_setup, mask_code = _mask(spec.mask_mod, spec.rownorm)
    prologue, rownorm_code, epilogue = _rownorm(spec.rownorm)
    score_code = "\n".join(f"            {line}" for line in score_code.splitlines())
    if not score_code:
        score_code = "            pass"
    key_len_line = "    sk = k.shape[-2]" if spec.mask_mod is None else ""

    causal_check = ""
    if spec.mask_mod is not None:
        causal_check = "    if q.shape[-2] != k.shape[-2]:\n        raise ValueError(\"causal attention requires Sq == Sk\")\n"

    return f'''import torch


TILE_Q = {tile_q}
TILE_K = {tile_k}


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    tensors = (q, k, v)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("q, k, and v must be 4D tensors")
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("q, k, and v must be CPU tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("q, k, and v must use torch.float32")
    if not (q.shape[:2] == k.shape[:2] == v.shape[:2]):
        raise ValueError("q, k, and v must have matching batch and head dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length")
{causal_check}


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _validate(q, k, v)

    seq_len = q.shape[-2]
{key_len_line}
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))

    for q_start in range(0, seq_len, TILE_Q):
        q_end = min(q_start + TILE_Q, seq_len)
        q_tile = q[..., q_start:q_end, :]
{prologue}
{mask_setup}

        for k_start in range(0, {k_stop}, TILE_K):
            k_end = min(k_start + TILE_K, {k_stop})
            k_tile = k[..., k_start:k_end, :]
            v_tile = v[..., k_start:k_end, :]
            scores = q_tile @ k_tile.transpose(-2, -1)
{score_code}
{mask_code}
{rownorm_code}

{epilogue}

    return out
'''


def _load_spec(module_name: str) -> AttentionSpec:
    module = importlib.import_module(module_name)
    spec = getattr(module, "spec", None)
    if not isinstance(spec, AttentionSpec):
        raise ValueError(f"{module_name}.spec must be an AttentionSpec")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiled Parallel attention")
    parser.add_argument("spec_module")
    parser.add_argument("--tile-q", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = generate(_load_spec(args.spec_module), args.tile_q, args.tile_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
