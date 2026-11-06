# One-hidden-layer neural network from scratch (MAT0611)

Individual course project for MAT0611 "Mathematics for Machine Learning" (Albert School).

Implementation, from scratch with NumPy, of a neural network with one hidden layer:
- `SimpleSLPClassifier` (binary and multi-class classification)
- `SimpleSLPRegressor` (regression)

Both follow the scikit-learn interface (`fit`, `predict`, `score`).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run the tests:
```bash
pytest -q
```

Open `report.ipynb` in VS Code or Jupyter to read the report. To execute it again from the command line, which takes
about 5 minutes:
```bash
pip install nbconvert
python -m nbconvert --to notebook --execute --inplace report.ipynb
```

Example:
```python
from slp_classifier import SimpleSLPClassifier

model = SimpleSLPClassifier(hidden_layer_size=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

## Main parameters

| Parameter | Default | Role |
|---|---|---|
| `hidden_layer_size` | `100` | Neurons in the hidden layer |
| `activation` | `"logistic"` | `"identity"`, `"logistic"`, `"tanh"` or `"relu"` |
| `learning_rate` | `0.001` | Learning rate |
| `max_iter` | `200` | Maximum number of epochs |
| `solver` | `"adam"` | `"adam"` or `"sgd"` |
| `batch_size` | `32` | Samples per update; `None` means full batch |
| `early_stopping` | `False` | Stop when the validation loss stops improving, and restore the best weights |
| `validation_fraction` | `0.1` | Share of the training data held out for early stopping |
| `n_iter_no_change` | `10` | Epochs without improvement allowed before stopping |
| `tol` | `1e-4` | Minimum decrease that counts as an improvement |
| `alpha` | `0.0` | L2 regularization strength |
| `random_state` | `None` | Random seed |

## Results

Mean test score over 5 seeds, each with its own 70/30 split. All networks have 100 hidden neurons and run for at most 200
epochs. R² is reported for diabetes, and accuracy for the other datasets.

| Dataset | Mine: defaults | Mine: sklearn-like Adam | sklearn: defaults | Mine: SGD | sklearn: SGD | Linear baseline |
|---|---:|---:|---:|---:|---:|---:|
| `load_diabetes` | 0.489 | 0.436 | 0.418 | 0.278 | 0.345 | 0.488 |
| `load_breast_cancer` | 0.968 | 0.970 | 0.973 | 0.926 | 0.937 | 0.980 |
| Titanic | 0.797 | 0.819 | 0.821 | 0.729 | 0.716 | 0.795 |
| `load_iris` | 0.933 | 0.938 | 0.933 | 0.622 | 0.569 | 0.938 |
| `load_digits` | 0.975 | 0.979 | 0.978 | 0.792 | 0.791 | 0.970 |

- "Mine: sklearn-like Adam" and "Mine: SGD" use the same settings as the two scikit-learn columns. The gaps between
  the implementations are smaller than the spread between seeds.
- SGD runs use `learning_rate=0.001` and no momentum. A larger learning rate narrows the gap with Adam: see section 5.4 of
  the report.
- The report also measures the effect of the batch size, early stopping, standardization and the optimizer.

**Test suite:** 107 tests pass, made up of the 54 provided tests and 53 additional tests in `test_gradients.py`.

**Not implemented:** the arbitrary-depth bonus (list of layer sizes).

## Structure

| File | Role |
|---|---|
| `activations.py` | Activation functions and their derivatives |
| `slp_base.py` | Shared base class: validation, initialization, optimizers, training loop, early stopping |
| `slp_regressor.py` | Regressor |
| `slp_classifier.py` | Classifier |
| `test_activations.py`, `test_models.py`, `test_smoke.py` | Tests provided with the project |
| `test_gradients.py` | Additional tests: finite-difference gradient checks and bonus features |
| `report.ipynb` | Report, executed with fixed seeds |
| `Data/titanic/train.csv` | Titanic data from Kaggle, used by the report |
| `neural_network_project.pdf` | Project statement |
