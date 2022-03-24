# 🌀: Various Machine learning models implemented using [JAX](https://jax.readthedocs.io/en/latest/index.html) and [Flax](https://github.com/google/flax). 🌀

## Models 

| Example | Code| Test |
|--- | --- | --- |
|**MLP**| [mlp.py](src/jax_examples/models/mlp.py)|[test_mlp.py](tests/test_mlp.py)|

## Examples

Train a MLP
```python
from sklearn.datasets import load_diabetes, load_digits
from jax_examples.models.mlp import MLP

# Regression
X, y = load_diabetes(return_X_y=True)
regression = MLP(task_type="regression")
regression.fit(X, y)
regression.predict(X)

# Classification
X, y = load_digits(return_X_y=True)
clf = MLP(task_type="classification")
clf.fit(X, y)
clf.predict(X)

```

## License

[MIT](LICENSE.md)
