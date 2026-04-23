# Search Space DSL

The `automl.search.space` module exposes a small, explicit DSL for declaring the joint
search space over preprocessing, feature selection, model family, and model
hyperparameters.

## Primitives

- `Float(name, low, high, *, log=False, step=None)`
- `Int(name, low, high, *, log=False, step=1)`
- `Categorical(name, choices)`
- `Conditional(when={parent_name: value_or_set}, node=...)`
- `SearchSpace([...])` — composes primitives + conditionals.

## Example

```python
from automl.search.space import Categorical, Float, Int, Conditional, SearchSpace

space = SearchSpace([
    Categorical("model", ["logreg", "rf"]),
    Conditional(
        when={"model": "logreg"},
        node=Float("logreg_C", 1e-3, 1e3, log=True),
    ),
    Conditional(
        when={"model": "rf"},
        node=Int("rf_n_estimators", 50, 1000, log=True),
    ),
])
```

A sample is a dict respecting the conditional gates:

```python
sample = space.sample(rng_or_optuna_trial)
# {"model": "rf", "rf_n_estimators": 384}
```

The DSL is intentionally tiny so a new contributor can grok it in five minutes.
