from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    name: str
    value: float | None = None


@dataclass(frozen=True)
class AttentionSpec:
    name: str
    pattern: str
    score_mod: tuple[Op, ...]
    mask_mod: Op | None
    rownorm: Op


def scale(value=None) -> Op:
    return Op("scale", value)


def relu() -> Op:
    return Op("relu")


def causal() -> Op:
    return Op("causal")


def softmax() -> Op:
    return Op("softmax")


def identity() -> Op:
    return Op("identity")
