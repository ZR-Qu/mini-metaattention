from miniattn.spec import AttentionSpec, causal, identity, relu, scale


spec = AttentionSpec(
    name="causal_relu",
    pattern="parallel",
    score_mod=(scale(), relu()),
    mask_mod=causal(),
    rownorm=identity(),
)
