from miniattn.spec import AttentionSpec, causal, scale, softmax


spec = AttentionSpec(
    name="causal_softmax",
    pattern="parallel",
    score_mod=(scale(),),
    mask_mod=causal(),
    rownorm=softmax(),
)
