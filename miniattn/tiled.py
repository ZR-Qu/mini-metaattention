import torch


def _validate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_q: int
) -> None:
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
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention requires Sq == Sk")
    if not isinstance(tile_q, int) or tile_q <= 0:
        raise ValueError("tile_q must be a positive integer")


def tiled(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_q: int
) -> torch.Tensor:
    """Causal attention tiled only along the query dimension."""
    _validate(q, k, v, tile_q)

    scale = q.shape[-1] ** -0.5
    seq_len = q.shape[-2]
    k_t = k.transpose(-2, -1)
    key_indices = torch.arange(seq_len, device=q.device)
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))

    for start in range(0, seq_len, tile_q):
        end = min(start + tile_q, seq_len)
        scores = q[..., start:end, :] @ k_t
        scores = scores * scale

        query_indices = torch.arange(start, end, device=q.device)
        causal = key_indices[None, :] > query_indices[:, None]
        scores = scores.masked_fill(causal, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        out[..., start:end, :] = probs @ v

    return out
