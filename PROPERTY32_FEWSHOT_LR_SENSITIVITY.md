# Property32 Few-Shot Learning-Rate Sensitivity

The primary twenty-shot study compares a low-learning-rate warm start
(`1e-5`) with a from-scratch control trained at `3e-4`. This post-hoc
sensitivity analysis fills the two missing cells:

| Initialization | 1e-5 | 3e-4 |
|---|---|---|
| Pretrained | Existing primary run | New sensitivity run |
| Random initialization | New sensitivity run | Existing primary control |

Every cell uses the same 20 support examples, 200 updates, architecture,
target properties, and full 5,000-example validation endpoint. It adds 120
high-learning-rate warm starts and 24 low-learning-rate random controls.

The analysis is explicitly exploratory and validation-only. The Property32
test split was already consumed by the frozen primary experiments and will not
be reused for model selection or sensitivity testing. Matched-learning-rate
pretrained-minus-random contrasts are first macro-averaged over four target
families and both pool directions, then summarized as mean plus/minus sample
standard deviation over the three joint task-split/model-seed replicates.

The machine-readable protocol is
[`configs/property32_fewshot_lr_sensitivity.toml`](configs/property32_fewshot_lr_sensitivity.toml).
