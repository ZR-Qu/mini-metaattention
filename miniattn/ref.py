import torch


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


def ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _validate(q, k, v)
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention requires Sq == Sk")

    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale

    seq_len = q.shape[-2]
    indices = torch.arange(seq_len, device=q.device)
    causal = indices[None, :] > indices[:, None]
    scores = scores.masked_fill(causal, float("-inf"))

    return torch.softmax(scores, dim=-1) @ v


def ref_relu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _validate(q, k, v)

    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1)
    scores = torch.relu(scores * scale)
    return scores @ v


def ref_causal_relu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    _validate(q, k, v)
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention requires Sq == Sk")

    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1)
    scores = torch.relu(scores * scale)

    seq_len = q.shape[-2]
    indices = torch.arange(seq_len, device=q.device)
    causal = indices[None, :] > indices[:, None]
    scores = scores.masked_fill(causal, 0.0)

    return scores @ v
