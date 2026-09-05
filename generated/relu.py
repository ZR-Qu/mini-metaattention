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



def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _validate(q, k, v)

    seq_len = q.shape[-2]
    sk = k.shape[-2]
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))

    for q_start in range(0, seq_len, TILE_Q):
        q_end = min(q_start + TILE_Q, seq_len)
        q_tile = q[..., q_start:q_end, :]
        o = q.new_zeros((*q_tile.shape[:-1], v.shape[-1]))


        for k_start in range(0, sk, TILE_K):
            k_end = min(k_start + TILE_K, sk)
            k_tile = k[..., k_start:k_end, :]
            v_tile = v[..., k_start:k_end, :]
            scores = q_tile @ k_tile.transpose(-2, -1)
            scores = scores * q.shape[-1] ** -0.5
            scores = torch.relu(scores)

            o = o + scores @ v_tile

        out[..., q_start:q_end, :] = o

    return out
