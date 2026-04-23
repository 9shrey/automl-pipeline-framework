# Extending AutoML

This guide will land alongside the registries. Placeholder with the three target stories:

1. **Add an estimator** — implement `EstimatorAdapter` and decorate with `@register_estimator("my_model")`.
2. **Add a feature selector** — register a factory in `automl.feature_selection.factory`.
3. **Add a sampler / pruner** — wire it through `automl.search.samplers` / `pruners`.

Concrete code samples will be filled in as those modules are implemented.
