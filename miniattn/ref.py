import torch


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    tensors = (q, k, v)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("q, k, and v must be 4D tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k, and v must have the same shape")
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("q, k, and v must be CPU tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("q, k, and v must use torch.float32")


def ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _validate(q, k, v)

    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale

    seq_len = q.shape[-2]
    indices = torch.arange(seq_len, device=q.device)
    causal = indices[None, :] > indices[:, None]
    scores = scores.masked_fill(causal, float("-inf"))

    return torch.softmax(scores, dim=-1) @ v
