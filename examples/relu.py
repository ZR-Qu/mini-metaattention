from miniattn.spec import AttentionSpec, identity, relu, scale


spec = AttentionSpec(
    name="relu",
    pattern="parallel",
    score_mod=(scale(), relu()),
    mask_mod=None,
    rownorm=identity(),
)
