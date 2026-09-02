import torch


def _validate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_q: int, tile_k: int
) -> None:
    tensors = (q, k, v)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("q, k, and v must be 4D tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k, and v must have the same shape")
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("q, k, and v must be CPU tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("q, k, and v must use torch.float32")
    if not isinstance(tile_q, int) or tile_q <= 0:
        raise ValueError("tile_q must be a positive integer")
    if not isinstance(tile_k, int) or tile_k <= 0:
        raise ValueError("tile_k must be a positive integer")


def online(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_q: int, tile_k: int
) -> torch.Tensor:
    """Causal attention tiled along Q and K with online softmax."""
    _validate(q, k, v, tile_q, tile_k)

    scale = q.shape[-1] ** -0.5
    seq_len = q.shape[-2]
    idx = torch.arange(seq_len, device=q.device)
    out = torch.empty_like(q)

    for q_start in range(0, seq_len, tile_q):
        q_end = min(q_start + tile_q, seq_len)
        q_tile = q[..., q_start:q_end, :]

        m = torch.full(
            (*q_tile.shape[:-1], 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        r = torch.zeros_like(m)
        o = torch.zeros_like(q_tile)
        q_idx = torch.arange(q_start, q_end, device=q.device)

        for k_start in range(0, seq_len, tile_k):
            k_end = min(k_start + tile_k, seq_len)
            k_tile = k[..., k_start:k_end, :]
            v_tile = v[..., k_start:k_end, :]

            scores = q_tile @ k_tile.transpose(-2, -1)
            scores = scores * scale

            k_idx = idx[k_start:k_end]
            causal = k_idx[None, :] > q_idx[:, None]
            scores = scores.masked_fill(causal, float("-inf"))

            block_max = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, block_max)
            o_scale = torch.exp(m - m_new)

            r = r * o_scale
            scores = torch.exp(scores - m_new)
            r = r + scores.sum(dim=-1, keepdim=True)
            o = o * o_scale + scores @ v_tile
            m = m_new

        out[..., q_start:q_end, :] = o / r

    return out
