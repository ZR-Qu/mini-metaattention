import torch


TILE_Q = 64
TILE_K = 64


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
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention requires Sq == Sk")



def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _validate(q, k, v)

    seq_len = q.shape[-2]

    out = q.new_empty((*q.shape[:-1], v.shape[-1]))

    for q_start in range(0, seq_len, TILE_Q):
        q_end = min(q_start + TILE_Q, seq_len)
        q_tile = q[..., q_start:q_end, :]
        m = torch.full(
            (*q_tile.shape[:-1], 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        r = torch.zeros_like(m)
        o = q.new_zeros((*q_tile.shape[:-1], v.shape[-1]))
        q_idx = torch.arange(q_start, q_end, device=q.device)

        for k_start in range(0, q_end, TILE_K):
            k_end = min(k_start + TILE_K, q_end)
            k_tile = k[..., k_start:k_end, :]
            v_tile = v[..., k_start:k_end, :]
            scores = q_tile @ k_tile.transpose(-2, -1)
            scores = scores * q.shape[-1] ** -0.5
            k_idx = torch.arange(k_start, k_end, device=q.device)
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
